#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KWS-CTC Gradio WebUI。

启动：
    /root/miniconda/envs/wekws/bin/python infer_webui.py --gpu 0

指定部署模型（例如 avg_30.pt）：
    /root/miniconda/envs/wekws/bin/python infer_webui.py --gpu 0 --checkpoint avg_30.pt

浏览器访问：
    http://127.0.0.1:7860
"""

import argparse
import json
import traceback
from pathlib import Path

import gradio as gr

from infer_core import KwsEngine, list_epoch_checkpoints, list_named_checkpoints


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

_engine = None


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


def run_detection(audio_path, keywords, checkpoint, dump_topk):
    if audio_path is None:
        return "请先上传音频或使用麦克风录音。", ""

    try:
        engine = get_engine(checkpoint)
        engine.set_keywords(keywords)
        result = engine.detect(audio_path, include_frame_topk=bool(dump_topk))
        log = json.dumps(result, ensure_ascii=False, indent=2)
        return format_result(result, show_topk=bool(dump_topk)), log
    except Exception:
        return "### 检测失败\n请查看下方日志。", traceback.format_exc()


def refresh_checkpoints(current):
    return gr.update(choices=checkpoint_choices(), value=current)


with gr.Blocks(title="KWS CTC 推理测试") as demo:
    gr.Markdown(
        "# KWS CTC 关键词检测\n\n"
        "上传音频或直接用麦克风录音，点击「开始检测」。\n"
        "模型常驻内存，首次加载后会更快。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="音频输入（上传 / 麦克风）",
                type="filepath",
                sources=["upload", "microphone"],
            )
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
            detect_btn = gr.Button("开始检测", variant="primary")

        with gr.Column(scale=1):
            result_output = gr.Markdown(label="检测结果")
            log_output = gr.Textbox(label="日志", lines=12, interactive=False)

    detect_btn.click(
        run_detection,
        inputs=[audio_input, keywords_input, checkpoint_input, topk_input],
        outputs=[result_output, log_output],
    )
    refresh_btn.click(
        refresh_checkpoints,
        inputs=[checkpoint_input],
        outputs=[checkpoint_input],
    )


if __name__ == "__main__":
    print(f"checkpoints: {checkpoint_choices()}")
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=args.port)
