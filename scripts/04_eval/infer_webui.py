#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KWS-CTC Gradio WebUI.

启动：
    source scripts/common/env.sh
    "$PYTHON_BIN" scripts/04_eval/infer_webui.py --gpu 0

指定部署模型（例如 avg_30.pt）：
    "$PYTHON_BIN" scripts/04_eval/infer_webui.py --gpu 0 --checkpoint avg_30.pt

浏览器访问：
    http://127.0.0.1:7860

后端说明：
- PyTorch：兼容原有的整段上传检测，checkpoint 可选。
- Android ONNX：直接使用 assets/kws.onnx + config.json；上传检测按流式会话跑，
  麦克风输入按块喂入并实时显示 live 事件，点击“结束并生成 final”后 flush。
"""

import argparse
import json
import math
import tempfile
import traceback
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from infer_core import KwsEngine, list_epoch_checkpoints, list_named_checkpoints
from onnx_kws_core import OnnxKwsRuntime


parser = argparse.ArgumentParser(description="KWS CTC WebUI")
parser.add_argument("--gpu", type=int, default=0, help="gpu id for inference")
parser.add_argument("--port", type=int, default=7860, help="server port")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="auto",
    help="default checkpoint; use 'avg_30.pt' for the averaged deployment model",
)
args = parser.parse_args()

BACKEND_PYTORCH = "PyTorch（整段检测）"
BACKEND_ONNX = "Android ONNX（部署口径，流式）"

_engine = None
_onnx_runtime = None
_streams = {}


def checkpoint_choices():
    choices = ["auto"]
    choices.extend(name for name, _ in list_named_checkpoints())
    choices.extend(p.name for _, p in reversed(list_epoch_checkpoints()))
    if args.checkpoint and args.checkpoint not in choices:
        choices.insert(1, args.checkpoint)
    return choices


def get_engine(checkpoint):
    global _engine
    if _engine is None:
        _engine = KwsEngine(checkpoint=checkpoint, gpu=args.gpu)
    else:
        _engine.set_checkpoint(checkpoint)
    return _engine


def get_onnx_runtime():
    global _onnx_runtime
    if _onnx_runtime is None:
        _onnx_runtime = OnnxKwsRuntime()
    return _onnx_runtime


def _format_token_items(items):
    if not items:
        return "-"
    return "; ".join(
        f"{p['token_id']}:{p['token']}={p['prob']:.6f}" for p in items
    )


def format_result(result, show_topk=False):
    lines = ["### 检测结果"]
    if not result["detected"]:
        lines.extend([
            "**未检测到关键词**",
            "",
            f"- checkpoint: `{Path(result['checkpoint']).name}`",
            f"- keywords: {result['keywords']}",
        ])
        return "\n".join(lines)

    lines.extend([
        f"**命中关键词：`{result['keyword']}`**",
        "",
        f"- 分数: `{result['score']}`",
        f"- 起止帧: `{result['start_frame']}` → `{result['end_frame']}`",
        f"- 持续帧数: `{result['duration']}`",
        f"- checkpoint: `{Path(result['checkpoint']).name}`",
        f"- keywords: {result['keywords']}",
        "",
        "#### 命中 token 概率",
        "| token | id | prob | frame | ms |",
        "|---|---:|---:|---:|---:|",
    ])
    for token in result.get("token_probs") or []:
        lines.append(
            f"| {token['token']} | {token['token_id']} | "
            f"{token['prob']:.6f} | {token['frame']} | {token['time_ms']} |"
        )

    frame_topk = result.get("frame_topk") or []
    if show_topk and frame_topk:
        max_display = 100
        lines.extend([
            "",
            f"#### 每帧 top-k（展示前 {min(max_display, len(frame_topk))} / "
            f"{len(frame_topk)} 帧，完整数据在日志）",
            "| frame | ms | raw top-k | kept |",
            "|---:|---:|---|---|",
        ])
        for frame in frame_topk[:max_display]:
            lines.append(
                f"| {frame['frame']} | {frame['time_ms']} | "
                f"{_format_token_items(frame['raw_topk'])} | "
                f"{_format_token_items(frame['kept'])} |"
            )

    return "\n".join(lines)


def _event_signature(event):
    return (f"{event['channel']}/{event['keyword']}/"
            f"{event['start_ms']}/{event['end_ms']}")


def _format_onnx_events(events, title="检测结果"):
    lines = [f"### {title}", ""]
    if not events:
        lines.append("**未检测到关键词**")
        return "\n".join(lines)
    lines.extend([
        "| 状态 | channel | 关键词 | score | 阈值 | 起止(ms) |",
        "|---|---:|---|---:|---:|---:|",
    ])
    for event in events:
        status = event.get("status", "live")
        lines.append(
            f"| {status} | {event['channel']} | `{event['keyword']}` | "
            f"{float(event['score']):.6f} | {float(event['threshold']):.6f} | "
            f"{event['start_ms']} → {event['end_ms']} |"
        )
    lines.append("")
    lines.append("score 口径：`sqrt(∏ token_prob)`；阈值来自每个关键词的 trigger_threshold。")
    return "\n".join(lines)


def _read_audio_file(audio_path):
    """Read an uploaded file or browser recording into int16-scale float32 mono PCM.

    soundfile covers wav/flac/ogg; browser microphone recordings are often
    webm/mp4, which Gradio decodes through pydub + ffmpeg instead.
    """
    path = Path(audio_path).expanduser()
    try:
        samples, rate = sf.read(str(path), dtype="int16", always_2d=True)
        values = samples[:, 0]
    except Exception:
        # Browser recordings are often webm/opus. pydub decodes them with a
        # 32-bit sample width; rescaling to 16-bit keeps the same amplitude
        # convention as soundfile's int16 output below.
        from pydub import AudioSegment

        audio = AudioSegment.from_file(str(path)).set_sample_width(2)
        rate = audio.frame_rate
        values = np.asarray(audio.get_array_of_samples())
        if audio.channels > 1:
            values = values.reshape(-1, audio.channels)[:, 0]
        values = np.clip(values, -32768, 32767)
    return np.asarray(values, dtype=np.float32).reshape(-1), int(rate)


def _resample_pcm(pcm, src_rate, dst_rate):
    """One-shot resample for static detection; 16 kHz input is returned as-is."""
    values = np.asarray(pcm, dtype=np.float32).reshape(-1)
    src_rate, dst_rate = int(src_rate), int(dst_rate)
    if src_rate == dst_rate or values.size == 0:
        return values
    import torch
    import torchaudio.functional as F

    resampled = F.resample(
        torch.from_numpy(values).unsqueeze(0), src_rate, dst_rate)
    return resampled.squeeze(0).numpy().astype(np.float32)


class _StreamResampler:
    """Stateful resampler so chunk boundaries do not lose samples.

    Browser microphones usually deliver 48 kHz while the deployed model needs
    16 kHz.  Resampling each chunk independently would break waveform
    continuity; keeping an input-side history and dropping its resampled
    prefix reproduces the full-signal resampling across chunk boundaries.
    """

    def __init__(self, src_rate, dst_rate=16000, history_ms=100):
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self.history_size = max(256, self.src_rate * int(history_ms) // 1000)
        self._buf = np.zeros(0, dtype=np.float32)

    def process(self, pcm):
        values = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return values
        if self.src_rate == self.dst_rate:
            return values
        joined = np.concatenate((self._buf, values))
        resampled = _resample_pcm(joined, self.src_rate, self.dst_rate)
        drop = math.ceil(self._buf.size * self.dst_rate / self.src_rate)
        self._buf = joined[-self.history_size:].copy()
        return resampled[drop:]


def _prepare_detection_path(audio_path):
    """Convert browser-only formats (webm/mp4/...) to a wav KwsEngine can read."""
    path = Path(audio_path).expanduser().resolve()
    try:
        sf.info(str(path))
        return str(path)
    except Exception:
        pcm, rate = _read_audio_file(path)
        out = Path(tempfile.gettempdir()) / f"kws_webui_{uuid.uuid4().hex}.wav"
        sf.write(str(out), pcm.astype(np.int16), rate, subtype="PCM_16")
        return str(out)


def _decode_onnx_wav(audio_path):
    runtime = get_onnx_runtime()
    config = runtime.config
    samples, rate = _read_audio_file(audio_path)
    resampler = _StreamResampler(rate, config.sample_rate)
    channel = 0
    chunk_samples = max(1, rate // 10)  # 100 ms at the source sample rate
    stream = runtime.new_stream(channel)
    for offset in range(0, samples.size, chunk_samples):
        pcm = resampler.process(samples[offset:offset + chunk_samples])
        if pcm.size == 0:
            continue
        stream.accept_pcm(pcm)
        runtime.decode_once([stream])
    stream.finish()
    runtime.decode_all([stream])
    events = []
    for event in stream.current_events():
        item = dict(event)
        item["score"] = round(float(event["score"]), 6)
        item["threshold"] = round(float(event["threshold"]), 6)
        events.append(item)
    return events


def run_detection(audio_path, keywords, checkpoint, dump_topk, backend):
    if audio_path is None:
        return "请先上传音频或使用麦克风录音。", ""

    try:
        if backend == BACKEND_ONNX:
            events = _decode_onnx_wav(audio_path)
            log = {
                "backend": "android-onnx",
                "onnx": str(get_onnx_runtime().config.onnx_path),
                "config_json": str(get_onnx_runtime().config.config_json),
                "events": events,
            }
            return (_format_onnx_events(events),
                    json.dumps(log, ensure_ascii=False, indent=2))

        engine = get_engine(checkpoint)
        engine.set_keywords(keywords)
        detection_path = _prepare_detection_path(audio_path)
        try:
            result = engine.detect(
                detection_path, include_frame_topk=bool(dump_topk))
        finally:
            if Path(detection_path).name.startswith("kws_webui_"):
                Path(detection_path).unlink(missing_ok=True)
        log = json.dumps(result, ensure_ascii=False, indent=2)
        return format_result(result, show_topk=bool(dump_topk)), log
    except Exception:
        return "### 检测失败\n请查看下方日志。", traceback.format_exc()


def refresh_checkpoints(current):
    return gr.update(choices=checkpoint_choices(), value=current)


_BACKEND_NOTE_ONNX = (
    "**当前：Android ONNX（部署口径）** · 直接加载 "
    "`assets/kws.onnx` + `config.json`；阈值 `嗨小问=0.303`、`你好问问=0.039`。"
)
_BACKEND_NOTE_PYTORCH = (
    "**当前：PyTorch（整段检测）** · 使用下方 checkpoint 与关键词做整段推理；"
    "麦克风流式仅支持 Android ONNX。"
)


def update_backend_layout(backend):
    """Only changes visibility/interactivity; inference logic is untouched."""
    is_onnx = backend == BACKEND_ONNX
    return (
        gr.update(value=_BACKEND_NOTE_ONNX if is_onnx else _BACKEND_NOTE_PYTORCH),
        gr.update(visible=not is_onnx),
        gr.update(interactive=is_onnx),
    )


def _normalize_stream_audio(audio):
    if isinstance(audio, tuple) and len(audio) == 2:
        rate, data = audio
    else:
        rate, data = 16000, audio
    values = np.asarray(data)
    if values.ndim == 2:
        values = values[:, 0] if values.shape[1] <= 8 else values.reshape(-1)
    else:
        values = values.reshape(-1)
    if values.size == 0:
        return int(rate), np.zeros(0, dtype=np.float32)
    # Gradio's microphone chunks are decoded by pydub: wav arrives as int16,
    # but webm/opus arrives as int32 (either 32-bit full scale or 16-bit
    # shifted left by 16). Normalize every integer width to the int16 scale
    # expected by the streaming frontend; floats are peak-normalized [-1, 1].
    if np.issubdtype(values.dtype, np.floating):
        scaled = values.astype(np.float64) * (1 << 15)
    elif values.dtype.itemsize == 1:
        scaled = values.astype(np.float64) * (1 << 8)
    elif values.dtype.itemsize == 2:
        scaled = values.astype(np.float64)
    elif values.dtype.itemsize == 4:
        scaled = values.astype(np.float64) / (1 << 16)
    else:
        scaled = values.astype(np.float64)
    pcm = np.clip(scaled, -32768, 32767).astype(np.float32)
    return int(rate), pcm


def _render_stream_events(sid):
    record = _streams.get(sid)
    status = ""
    if record is not None and record.get("chunks"):
        status = (
            f"\n\n输入状态：已接收 {record['chunks']} 块 · "
            f"源采样率 {record.get('rate')} Hz → 16 kHz · "
            f"峰值 {record.get('pcm_absmax')}。"
        )
    if record is None or not record["events"]:
        return "### 流式事件\n\n等待关键词命中…" + status
    lines = [
        "### 流式事件",
        "",
        "| 状态 | channel | 关键词 | score | 阈值 | 起止(ms) |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for _signature, (event, final) in record["events"].items():
        lines.append(
            f"| {'final' if final else 'live'} | {event['channel']} | "
            f"`{event['keyword']}` | {float(event['score']):.6f} | "
            f"{float(event['threshold']):.6f} | "
            f"{event['start_ms']} → {event['end_ms']} |"
        )
    lines.append("")
    lines.append("同一 `channel/keyword/startMs/endMs` 只刷新一行；停止后请点击“结束并生成 final”。")
    if status:
        lines.append(status)
    return "\n".join(lines)


def _update_stream_map(record, final=False):
    for hit in record["session"].current_events():
        event = dict(hit)
        event["score"] = round(float(hit["score"]), 6)
        event["threshold"] = round(float(hit["threshold"]), 6)
        signature = _event_signature(event)
        previous = record["events"].get(signature)
        if previous is not None and previous[1]:
            continue
        record["events"][signature] = (event, final)
    if final:
        for signature, (event, _) in list(record["events"].items()):
            record["events"][signature] = (event, True)


def _session_sid(sid, request):
    """Prefer Gradio's per-browser session hash over State round-trips.

    Streaming events can cancel earlier invocations (and with them their State
    updates), which would otherwise create a brand-new stream for every audio
    chunk and never accumulate enough audio for a keyword.
    """
    if request is not None:
        session_hash = getattr(request, "session_hash", None)
        if session_hash:
            return session_hash
    return sid or uuid.uuid4().hex


def stream_audio(audio, sid, backend, request: gr.Request = None):
    sid = _session_sid(sid, request)
    if backend != BACKEND_ONNX:
        return (sid,
                "### 流式事件\n\n流式输入仅支持 `Android ONNX（部署口径，流式）` 后端。",
                "")
    if audio is None:
        return sid, _render_stream_events(sid), ""

    try:
        rate, pcm = _normalize_stream_audio(audio)
        record = _streams.get(sid)
        if record is None:
            runtime = get_onnx_runtime()
            record = {
                "session": runtime.new_stream(0),
                "events": {},
                "resampler": _StreamResampler(rate, runtime.config.sample_rate),
            }
            _streams[sid] = record

        resampler = record["resampler"]
        if rate != resampler.src_rate:
            # Browser may change the callback rate mid-stream; restart the
            # resampler rather than feeding wrong-rate samples to the model.
            resampler = _StreamResampler(rate, get_onnx_runtime().config.sample_rate)
            record["resampler"] = resampler
        pcm = resampler.process(pcm)
        record["session"].accept_pcm(pcm)
        get_onnx_runtime().decode_once([record["session"]])
        _update_stream_map(record, final=False)
        absmax = round(float(np.max(np.abs(pcm))), 1) if pcm.size else 0
        record["chunks"] = record.get("chunks", 0) + 1
        record["rate"] = rate
        record["pcm_absmax"] = absmax
        log = {
            "sid": sid,
            "rate": rate,
            "pcm_absmax": absmax,
            "queued_frames": record["session"].num_queued_frames(),
            "events": record["session"].current_events(),
        }
        return (sid, _render_stream_events(sid),
                json.dumps(log, ensure_ascii=False, indent=2))
    except Exception:
        return sid, "### 流式事件\n\n流式处理失败，请查看日志。", traceback.format_exc()


def finalize_stream(sid, request: gr.Request = None):
    sid = _session_sid(sid, request)
    if not sid or sid not in _streams:
        return "### 流式事件\n\n没有可结束的流式会话。", ""
    try:
        record = _streams[sid]
        runtime = get_onnx_runtime()
        record["session"].finish()
        runtime.decode_all([record["session"]])
        _update_stream_map(record, final=True)
        log = {
            "sid": sid,
            "frames": record["session"].frame_offset,
            "events": record["session"].current_events(),
            "triggered_events": record["session"].triggered_events,
        }
        return (_render_stream_events(sid),
                json.dumps(log, ensure_ascii=False, indent=2, default=float))
    except Exception:
        return "### 流式事件\n\nfinal flush 失败，请查看日志。", traceback.format_exc()


def clear_stream(sid, request: gr.Request = None):
    sid = _session_sid(sid, request)
    _streams.pop(sid, None)
    return None, "### 流式事件\n\n等待关键词命中…", ""


with gr.Blocks(title="KWS CTC 推理测试") as demo:
    gr.Markdown(
        "# KWS CTC 关键词检测\n\n"
        "用「上传检测」跑整段音频，用「麦克风流式」观察 live/final 事件。"
        "模型常驻内存，首次加载后会更快。"
    )

    stream_state = gr.State(value=None)
    with gr.Row():
        with gr.Column(scale=1):
            backend_input = gr.Dropdown(
                label="推理后端（全局）",
                choices=[BACKEND_ONNX, BACKEND_PYTORCH],
                value=BACKEND_ONNX,
            )
        with gr.Column(scale=3):
            backend_note = gr.Markdown(value=_BACKEND_NOTE_ONNX)

    with gr.Tabs():
        with gr.Tab(label="上传检测"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="上传音频 / 录音（录完点“开始检测”）",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    detect_btn = gr.Button("开始检测", variant="primary")
                    with gr.Accordion(
                        "PyTorch 设置（关键词 / checkpoint / top-k）",
                        open=False,
                        visible=False,
                    ) as pytorch_settings:
                        keywords_input = gr.Textbox(
                            label="关键词（逗号分隔）",
                            value="嗨小问,你好问问",
                        )
                        checkpoint_input = gr.Dropdown(
                            label="Checkpoint（可选 avg_30.pt 或任意路径）",
                            choices=checkpoint_choices(),
                            value=args.checkpoint,
                            allow_custom_value=True,
                        )
                        refresh_btn = gr.Button("刷新 checkpoint 列表")
                        topk_input = gr.Checkbox(
                            label="输出每帧 top-k（调试，日志会很长）",
                            value=False,
                        )

                with gr.Column(scale=2):
                    result_output = gr.Markdown(label="检测结果")
                    with gr.Accordion("运行日志", open=False):
                        log_output = gr.Textbox(
                            label="日志", lines=12, interactive=False)

        with gr.Tab(label="麦克风流式"):
            gr.Markdown(
                "选择 `Android ONNX` 后端后，下方麦克风按块送入同一个流式会话；"
                "录音中实时刷新 live 事件。停止录音后点击「结束并生成 final」执行 flush。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    stream_audio_input = gr.Audio(
                        label="麦克风流式输入（自动转 16 kHz，仅 ONNX）",
                        type="numpy",
                        sources=["microphone"],
                        streaming=True,
                    )
                    with gr.Row():
                        finalize_btn = gr.Button("结束并生成 final")
                        clear_btn = gr.Button("清空流式结果")
                with gr.Column(scale=1):
                    stream_result = gr.Markdown(label="流式事件")
                    with gr.Accordion("流式日志", open=False):
                        stream_log = gr.Textbox(
                            label="流式日志", lines=10, interactive=False)

    detect_btn.click(
        run_detection,
        inputs=[audio_input, keywords_input, checkpoint_input, topk_input,
                backend_input],
        outputs=[result_output, log_output],
    )
    refresh_btn.click(
        refresh_checkpoints,
        inputs=[checkpoint_input],
        outputs=[checkpoint_input],
    )
    backend_input.change(
        update_backend_layout,
        inputs=[backend_input],
        outputs=[backend_note, pytorch_settings, stream_audio_input],
    )
    stream_audio_input.stream(
        stream_audio,
        inputs=[stream_audio_input, stream_state, backend_input],
        outputs=[stream_state, stream_result, stream_log],
        # Do not cancel in-flight chunks: with always_last, cancelled events
        # drop their State updates and every chunk starts a fresh session.
        trigger_mode="multiple",
        concurrency_limit=1,
    )
    finalize_btn.click(
        finalize_stream,
        inputs=[stream_state],
        outputs=[stream_result, stream_log],
    )
    clear_btn.click(
        clear_stream,
        inputs=[stream_state],
        outputs=[stream_state, stream_result, stream_log],
    )


if __name__ == "__main__":
    print(f"checkpoints: {checkpoint_choices()}")
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=args.port)
