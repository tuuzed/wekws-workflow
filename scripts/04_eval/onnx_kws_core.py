#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python ONNX streaming KWS core aligned with the Android Native engine.

The deployment assets are loaded directly from android/app/src/main/assets:
``kws.onnx`` is never converted to ``.ort`` and its posterior output is used
as-is (no second softmax).  One ORT session is shared by all streams; every
stream owns an independent frontend, cache, incremental CTC beam, event state
and cooldown timestamps.
"""

import argparse
import json
import math
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPTS_ROOT.parent
ASSETS_ROOT = WORKSPACE_ROOT / "android" / "app" / "src" / "main" / "assets"
DEFAULT_ONNX = ASSETS_ROOT / "kws.onnx"
DEFAULT_CONFIG_JSON = ASSETS_ROOT / "config.json"


@dataclass(frozen=True)
class KeywordConfig:
    text: str
    chars: Tuple[str, ...]
    tokens: Tuple[int, ...]
    trigger_threshold: float


@dataclass(frozen=True)
class RuntimeConfig:
    onnx_path: Path
    config_json: Path
    sample_rate: int
    feature_dim: int
    frame_length_ms: int
    frame_shift_ms: int
    hidden_dim: int
    cache_len: int
    vocab_size: int
    blank_id: int
    keywords: Tuple[KeywordConfig, ...]
    keywords_token_set: Tuple[int, ...]
    score_beam_size: int
    path_beam_size: int
    prob_threshold: float
    isclose_abs_tol: float
    cooldown_ms: int
    reset_decoder_after_trigger: bool
    reset_cache_after_trigger: bool
    chunk_frames: int
    intra_op_threads: int
    inter_op_threads: int

    @property
    def frame_length_samples(self) -> int:
        return self.sample_rate * self.frame_length_ms // 1000

    @property
    def frame_shift_samples(self) -> int:
        return self.sample_rate * self.frame_shift_ms // 1000


def _as_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def load_runtime_config(config_json=None, onnx_path=None) -> RuntimeConfig:
    config_path = Path(config_json or DEFAULT_CONFIG_JSON).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"runtime config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fin:
        raw = json.load(fin)

    if not isinstance(raw, dict):
        raise ValueError("runtime config root must be an object")
    ctc_raw = raw.get("ctc") or {}
    if "trigger_threshold" in ctc_raw:
        raise ValueError("global ctc.trigger_threshold is not allowed; use per-keyword thresholds")

    keywords = []
    raw_keywords = raw.get("keywords")
    if not isinstance(raw_keywords, list) or not raw_keywords:
        raise ValueError("runtime config must contain a non-empty keywords array")
    for item in raw_keywords:
        if not isinstance(item, dict):
            raise ValueError("each keyword must be an object")
        text = item.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("keyword text must be a non-empty string")
        if "trigger_threshold" not in item:
            raise ValueError(f"keyword '{text}' missing trigger_threshold")
        threshold = _as_number(item["trigger_threshold"], f"keyword '{text}' trigger_threshold")
        if threshold < 0.0:
            raise ValueError(f"keyword '{text}' has negative trigger_threshold")
        raw_tokens = item.get("tokens")
        if not isinstance(raw_tokens, list) or not raw_tokens:
            raise ValueError(f"keyword '{text}' must have a non-empty tokens array")
        tokens = tuple(_as_int(t, f"keyword '{text}' token") for t in raw_tokens)
        raw_chars = item.get("chars") or list(text)
        if not isinstance(raw_chars, list):
            raise ValueError(f"keyword '{text}' chars must be an array")
        keywords.append(KeywordConfig(text, tuple(str(c) for c in raw_chars), tokens, threshold))

    token_set = tuple(sorted({0} | {t for kw in keywords for t in kw.tokens}))

    model_path = Path(onnx_path or raw.get("model_path") or DEFAULT_ONNX.name)
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    model_path = model_path.expanduser().resolve()

    return RuntimeConfig(
        onnx_path=model_path,
        config_json=config_path,
        sample_rate=_as_int(raw.get("sample_rate", 16000), "sample_rate"),
        feature_dim=_as_int(raw.get("feature_dim", 40), "feature_dim"),
        frame_length_ms=_as_int(raw.get("frame_length_ms", 25), "frame_length_ms"),
        frame_shift_ms=_as_int(raw.get("frame_shift_ms", 10), "frame_shift_ms"),
        hidden_dim=_as_int(raw.get("hidden_dim", 256), "hidden_dim"),
        cache_len=_as_int(raw.get("cache_len", 105), "cache_len"),
        vocab_size=_as_int(raw.get("vocab_size", 2599), "vocab_size"),
        blank_id=_as_int(raw.get("blank_id", 0), "blank_id"),
        keywords=tuple(keywords),
        keywords_token_set=token_set,
        score_beam_size=_as_int(ctc_raw.get("score_beam_size", 3), "score_beam_size"),
        path_beam_size=_as_int(ctc_raw.get("path_beam_size", 20), "path_beam_size"),
        prob_threshold=_as_number(ctc_raw.get("prob_threshold", 0.05), "prob_threshold"),
        isclose_abs_tol=_as_number(ctc_raw.get("isclose_abs_tol", 1e-6), "isclose_abs_tol"),
        cooldown_ms=_as_int(ctc_raw.get("cooldown_ms", 1000), "cooldown_ms"),
        reset_decoder_after_trigger=bool(
            ctc_raw.get("reset_decoder_after_trigger", True)),
        reset_cache_after_trigger=bool(
            ctc_raw.get("reset_cache_after_trigger", False)),
        chunk_frames=_as_int((raw.get("runtime") or {}).get("chunk_frames", 80),
                             "chunk_frames"),
        intra_op_threads=_as_int((raw.get("runtime") or {}).get("intra_op_threads", 1),
                                 "intra_op_threads"),
        inter_op_threads=_as_int((raw.get("runtime") or {}).get("inter_op_threads", 1),
                                 "inter_op_threads"),
    )


def _is_dynamic_dim(dim: Any) -> bool:
    return not isinstance(dim, int) or dim <= 0


def validate_onnx_shapes(session: ort.InferenceSession, config: RuntimeConfig) -> None:
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}
    for name in ("input", "cache"):
        if name not in inputs:
            raise ValueError(f"ONNX model is missing input '{name}'")
    for name in ("output", "r_cache"):
        if name not in outputs:
            raise ValueError(f"ONNX model is missing output '{name}'")

    feat = inputs["input"].shape
    cache = inputs["cache"].shape
    if len(feat) != 3 or not _is_dynamic_dim(feat[0]) or not _is_dynamic_dim(feat[1]):
        raise ValueError("ONNX input 'input' must have dynamic N and T dimensions")
    if feat[2] != config.feature_dim:
        raise ValueError(
            f"ONNX input feature dim {feat[2]} != config {config.feature_dim}")
    if len(cache) != 3 or not _is_dynamic_dim(cache[0]):
        raise ValueError("ONNX input 'cache' must have dynamic batch dimension")
    if cache[1] != config.hidden_dim or cache[2] != config.cache_len:
        raise ValueError(
            f"ONNX cache shape {cache[1:]} != config "
            f"({config.hidden_dim}, {config.cache_len})")

    prob = outputs["output"].shape
    r_cache = outputs["r_cache"].shape
    if len(prob) != 3 or not _is_dynamic_dim(prob[0]) or not _is_dynamic_dim(prob[1]):
        raise ValueError("ONNX output 'output' must have dynamic N and T dimensions")
    if prob[2] != config.vocab_size:
        raise ValueError(
            f"ONNX output vocab size {prob[2]} != config {config.vocab_size}")
    if len(r_cache) != 3 or not _is_dynamic_dim(r_cache[0]):
        raise ValueError("ONNX output 'r_cache' must have dynamic batch dimension")


@dataclass
class _BeamNode:
    token: int
    frame: int
    prob: float


@dataclass
class _BeamHyp:
    prefix: Tuple[int, ...] = ()
    pb: float = 0.0
    pnb: float = 0.0
    nodes: List[_BeamNode] = field(default_factory=list)


def _is_sublist(main: Sequence[int], check: Sequence[int]) -> int:
    """Mirror wekws score_ctc.py / Native CtcBeamSearch::IsSublist exactly."""
    if len(main) < len(check):
        return -1
    if len(main) == len(check):
        return 0 if tuple(main) == tuple(check) else -1
    for i in range(len(main) - len(check)):
        if all(main[i + j] == check[j] for j in range(len(check))):
            return i
    return -1


class _CtcPrefixBeamSearch:
    """Incremental port of Native CtcBeamSearch / upstream ctc_prefix_beam_search."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.token_set = {config.blank_id}
        for keyword in config.keywords:
            self.token_set.update(keyword.tokens)
        self.reset()

    def reset(self) -> None:
        self.hyps: List[_BeamHyp] = [_BeamHyp(pb=1.0)]

    @staticmethod
    def _isclose_to_zero(value: float, abs_tol: float) -> bool:
        return abs(value) <= abs_tol

    def process_frame(self, probs: np.ndarray, frame: int) -> None:
        top = sorted(range(len(probs)),
                     key=lambda i: (-float(probs[i]), i))[:self.config.score_beam_size]
        kept = [(i, float(probs[i])) for i in top
                if float(probs[i]) > self.config.prob_threshold
                and i in self.token_set]
        if not kept:
            return

        entries: List[_BeamHyp] = []
        index: Dict[Tuple[int, ...], int] = {}

        def get_or_create(prefix: Tuple[int, ...]) -> _BeamHyp:
            found = index.get(prefix)
            if found is not None:
                return entries[found]
            created = _BeamHyp(prefix=prefix)
            index[prefix] = len(entries)
            entries.append(created)
            return created

        blank = self.config.blank_id
        abs_tol = self.config.isclose_abs_tol
        for token, prob in kept:
            for cur in self.hyps:
                last = cur.prefix[-1] if cur.prefix else -1
                if token == blank:
                    nxt = get_or_create(cur.prefix)
                    nxt.nodes = list(cur.nodes)
                    nxt.pb += cur.pb * prob + cur.pnb * prob
                elif token == last:
                    if not self._isclose_to_zero(cur.pnb, abs_tol):
                        nxt = get_or_create(cur.prefix)
                        nxt.nodes = list(cur.nodes)
                        nxt.pnb += cur.pnb * prob
                        if nxt.nodes and prob > nxt.nodes[-1].prob:
                            nxt.nodes[-1] = _BeamNode(token, frame, prob)
                    if not self._isclose_to_zero(cur.pb, abs_tol):
                        prefix = cur.prefix + (token,)
                        nxt = get_or_create(prefix)
                        nxt.nodes = list(cur.nodes)
                        nxt.nodes.append(_BeamNode(token, frame, prob))
                        nxt.pnb += cur.pb * prob
                else:
                    prefix = cur.prefix + (token,)
                    nxt = get_or_create(prefix)
                    if nxt.nodes:
                        if prob > nxt.nodes[-1].prob:
                            nxt.nodes[-1] = _BeamNode(token, frame, prob)
                    else:
                        nxt.nodes = list(cur.nodes)
                        nxt.nodes.append(_BeamNode(token, frame, prob))
                    nxt.pnb += cur.pb * prob + cur.pnb * prob

        # Python's sorted() is stable, matching Native stable_sort by score.
        entries.sort(key=lambda hyp: -(hyp.pb + hyp.pnb))
        self.hyps = entries[:self.config.path_beam_size]

    def find_best_hits(self) -> List[Optional[Dict[str, Any]]]:
        hits: List[Optional[Dict[str, Any]]] = [None] * len(self.config.keywords)
        for hyp in self.hyps:
            for idx, keyword in enumerate(self.config.keywords):
                if hits[idx] is not None:
                    continue
                offset = _is_sublist(hyp.prefix, keyword.tokens)
                if offset < 0 or offset + len(keyword.tokens) > len(hyp.nodes):
                    continue
                score = 1.0
                for node in hyp.nodes[offset:offset + len(keyword.tokens)]:
                    score *= node.prob
                token_probs = []
                for pos, node in enumerate(hyp.nodes[offset:offset + len(keyword.tokens)]):
                    token_probs.append({
                        "token": keyword.chars[pos] if pos < len(keyword.chars)
                        else str(node.token),
                        "token_id": node.token,
                        "frame": node.frame,
                        "prob": node.prob,
                    })
                hits[idx] = {
                    "keyword": keyword.text,
                    "score": math.sqrt(score),
                    "start_frame": hyp.nodes[offset].frame,
                    "end_frame": hyp.nodes[offset + len(keyword.tokens) - 1].frame,
                    "token_probs": token_probs,
                }
        return hits


class _StreamingFbank:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.remainder = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self.remainder = np.zeros(0, dtype=np.float32)

    def accept(self, pcm: np.ndarray) -> None:
        if self.config.sample_rate != 16000:
            raise ValueError("streaming fbank currently requires 16 kHz audio")
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        self.remainder = np.concatenate((self.remainder, samples))
        frame_length = self.config.frame_length_samples
        shift = self.config.frame_shift_samples
        if self.remainder.size < frame_length:
            return
        num_frames = 1 + (self.remainder.size - frame_length) // shift
        self._pending = num_frames
        # Frames are computed lazily in read(); keep the overlapping tail just
        # like Native FeaturePipeline so chunk boundaries do not skip frames.
        self._available = num_frames

    def num_queued_frames(self) -> int:
        return getattr(self, "_available", 0)

    def read(self, num_frames: int) -> np.ndarray:
        available = self.num_queued_frames()
        if available < num_frames:
            raise RuntimeError("not enough queued fbank frames")
        frame_length = self.config.frame_length_samples
        shift = self.config.frame_shift_samples
        windows = np.stack([
            self.remainder[i * shift:i * shift + frame_length]
            for i in range(num_frames)
        ])
        # kaldi.fbank() treats a 2-D tensor as (channel, samples), so compute
        # each pre-sliced 25 ms window independently to keep streaming frames
        # identical to the offline reference frontend.
        frames = []
        for i in range(num_frames):
            feat = kaldi.fbank(
                torch.from_numpy(windows[i]).unsqueeze(0),
                num_mel_bins=self.config.feature_dim,
                frame_length=self.config.frame_length_ms,
                frame_shift=self.config.frame_shift_ms,
                dither=0.0,
                energy_floor=0.0,
                sample_frequency=self.config.sample_rate,
            )
            frames.append(feat.squeeze(0).numpy())
        consumed = num_frames * shift
        self.remainder = self.remainder[consumed:].copy()
        self._available = available - num_frames
        return np.stack(frames).astype(np.float32)


class StreamSession:
    """Independent streaming state for one browser session / audio channel."""

    def __init__(self, runtime: "OnnxKwsRuntime", channel: int = 0):
        self.runtime = runtime
        self.channel = int(channel)
        config = runtime.config
        self.frontend = _StreamingFbank(config)
        self.decoder = _CtcPrefixBeamSearch(config)
        self.cache = np.zeros((config.hidden_dim, config.cache_len), dtype=np.float32)
        self.frame_offset = 0
        self.finished = False
        self.last_trigger_ms = [-1] * len(config.keywords)
        self.best_hits: List[Optional[Dict[str, Any]]] = [None] * len(config.keywords)
        self.has_hit = [False] * len(config.keywords)
        self.triggered_events: List[Dict[str, Any]] = []
        self.probs: List[np.ndarray] = []

    def accept_pcm(self, pcm: np.ndarray) -> None:
        if self.finished:
            raise RuntimeError("cannot accept PCM after finish()")
        self.frontend.accept(pcm)

    def accept_pcm16(self, pcm: np.ndarray) -> None:
        self.accept_pcm(np.asarray(pcm, dtype=np.int16).astype(np.float32))

    def accept_float_audio(self, audio: np.ndarray) -> None:
        """Accept float waveform normalized to [-1, 1], as returned by browsers."""
        values = np.asarray(audio, dtype=np.float32).reshape(-1)
        self.accept_pcm(np.clip(values * (1 << 15), -32768, 32767))

    def finish(self) -> None:
        self.finished = True

    def reset(self) -> None:
        self.frontend.reset()
        self.decoder.reset()
        self.cache.fill(0.0)
        self.frame_offset = 0
        self.finished = False
        self.last_trigger_ms = [-1] * len(self.runtime.config.keywords)
        self.best_hits = [None] * len(self.runtime.config.keywords)
        self.has_hit = [False] * len(self.runtime.config.keywords)
        self.triggered_events.clear()
        self.probs.clear()

    def num_queued_frames(self) -> int:
        return self.frontend.num_queued_frames()

    def _public_event(self, idx: int, hit: Dict[str, Any]) -> Dict[str, Any]:
        config = self.runtime.config
        event = dict(hit)
        event.update({
            "channel": self.channel,
            "start_ms": hit["start_frame"] * config.frame_shift_ms,
            "end_ms": hit["end_frame"] * config.frame_shift_ms,
            "threshold": config.keywords[idx].trigger_threshold,
        })
        return event

    def current_events(self) -> List[Dict[str, Any]]:
        config = self.runtime.config
        events = []
        for idx, hit in enumerate(self.best_hits):
            if not self.has_hit[idx] or hit is None:
                continue
            if hit["score"] < config.keywords[idx].trigger_threshold:
                continue
            events.append(self._public_event(idx, hit))
        return events


class OnnxKwsRuntime:
    """One shared ORT session with serialized decode calls."""

    def __init__(self, config_json=None, onnx_path=None, capture_probs=False,
                 baseline=False):
        self.config = load_runtime_config(config_json, onnx_path)
        if baseline:
            # Alignment-only mode: keep the legacy offline score anchor while
            # the deployed runtime still applies cooldown/reset immediately.
            self.config = replace(
                self.config,
                cooldown_ms=0,
                reset_decoder_after_trigger=False,
                reset_cache_after_trigger=False,
            )
        if not self.config.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.config.onnx_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.config.intra_op_threads
        options.inter_op_num_threads = self.config.inter_op_threads
        self.session = ort.InferenceSession(
            str(self.config.onnx_path), sess_options=options,
            providers=["CPUExecutionProvider"])
        validate_onnx_shapes(self.session, self.config)
        self.capture_probs = bool(capture_probs)
        self.lock = threading.RLock()

    def new_stream(self, channel: int = 0) -> StreamSession:
        return StreamSession(self, channel)

    def decode_once(self, streams: Sequence[StreamSession]) -> bool:
        with self.lock:
            queued = [s.num_queued_frames() for s in streams]
            max_queued = max(queued, default=0)
            if max_queued <= 0:
                return False
            frames = min(self.config.chunk_frames, max_queued)
            active = [s for s, q in zip(streams, queued) if q >= frames]
            if not active:
                return False
            self._decode_batch(active, frames)
            return True

    def decode_all(self, streams: Sequence[StreamSession]) -> None:
        with self.lock:
            while self.decode_once(streams):
                pass
            # Flush partial tail frames of finished streams.
            pending = [s for s in streams if s.finished and s.num_queued_frames() > 0]
            if pending:
                frames = max(s.num_queued_frames() for s in pending)
                active = [s for s in pending if s.num_queued_frames() == frames]
                self._decode_batch(active, frames)

    def _decode_batch(self, active: Sequence[StreamSession], frames: int) -> None:
        config = self.config
        batch = len(active)
        feats_flat = np.zeros((batch, frames, config.feature_dim), dtype=np.float32)
        cache_flat = np.zeros((batch, config.hidden_dim, config.cache_len), dtype=np.float32)
        for b, stream in enumerate(active):
            feats_flat[b] = stream.frontend.read(frames)
            cache_flat[b] = stream.cache

        outputs = self.session.run(None, {
            "input": feats_flat,
            "cache": cache_flat,
        })
        probs_batch = outputs[0]
        cache_batch = outputs[1]

        for b, stream in enumerate(active):
            stream.cache = np.array(cache_batch[b], dtype=np.float32, copy=True)
            if self.capture_probs:
                stream.probs.append(np.array(probs_batch[b], dtype=np.float32, copy=True))
            triggered = False
            for t in range(frames):
                frame = stream.frame_offset + t
                stream.decoder.process_frame(probs_batch[b, t], frame)
                hits = stream.decoder.find_best_hits()
                for idx, hit in enumerate(hits):
                    if hit is None:
                        continue
                    keyword = config.keywords[idx]
                    if hit["score"] < keyword.trigger_threshold:
                        continue
                    now_ms = hit["end_frame"] * config.frame_shift_ms
                    last_ms = stream.last_trigger_ms[idx]
                    if (config.cooldown_ms > 0 and last_ms >= 0
                            and now_ms - last_ms < config.cooldown_ms):
                        continue
                    stream.best_hits[idx] = hit
                    stream.has_hit[idx] = True
                    stream.last_trigger_ms[idx] = now_ms
                    stream.triggered_events.append(stream._public_event(idx, hit))
                    triggered = True
                if triggered:
                    if config.reset_decoder_after_trigger:
                        stream.decoder.reset()
                    if config.reset_cache_after_trigger:
                        stream.cache.fill(0.0)
                    triggered = False
            stream.frame_offset += frames


def _event_to_json(event: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(event)
    result["score"] = round(float(event["score"]), 6)
    result["threshold"] = round(float(event["threshold"]), 6)
    result["token_probs"] = [
        {**item, "prob": round(float(item["prob"]), 6)}
        for item in event.get("token_probs", [])
    ]
    return result


def _read_wav(wav_path: str, sample_rate: int, channel: int) -> np.ndarray:
    pcm, rate = sf.read(wav_path, dtype="int16", always_2d=True)
    if rate != sample_rate:
        raise ValueError(f"expected {sample_rate} Hz, got {rate}")
    if channel < 0 or channel >= pcm.shape[1]:
        raise ValueError(f"channel {channel} out of range for {pcm.shape[1]} channel wav")
    return pcm[:, channel]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_json", default=str(DEFAULT_CONFIG_JSON))
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--list_keywords", action="store_true",
                        help="print keyword tokens/thresholds and exit")
    parser.add_argument("--wav", help="WAV file to decode as one streaming session")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--chunk_ms", type=int, default=100,
                        help="PCM chunk size for streaming simulation")
    parser.add_argument("--baseline", action="store_true",
                        help="disable cooldown/reset for legacy score alignment")
    parser.add_argument("--probs_out", help="dump concatenated posterior matrix to .npy")
    parser.add_argument("--summary_json", help="write the final event summary as JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime = OnnxKwsRuntime(
        config_json=args.config_json,
        onnx_path=args.onnx,
        capture_probs=bool(args.probs_out),
        baseline=args.baseline,
    )
    config = runtime.config

    if args.list_keywords:
        print(json.dumps({
            "onnx": str(config.onnx_path),
            "config_json": str(config.config_json),
            "cooldown_ms": config.cooldown_ms,
            "reset_decoder_after_trigger": config.reset_decoder_after_trigger,
            "reset_cache_after_trigger": config.reset_cache_after_trigger,
            "keywords": [
                {
                    "text": kw.text,
                    "chars": list(kw.chars),
                    "tokens": list(kw.tokens),
                    "trigger_threshold": kw.trigger_threshold,
                }
                for kw in config.keywords
            ],
        }, ensure_ascii=False, indent=2))
        return

    if not args.wav:
        raise SystemExit("provide --wav or --list_keywords")
    pcm = _read_wav(args.wav, config.sample_rate, args.channel)
    chunk_samples = max(1, config.sample_rate * args.chunk_ms // 1000)
    stream = runtime.new_stream(args.channel)
    for offset in range(0, pcm.size, chunk_samples):
        stream.accept_pcm16(pcm[offset:offset + chunk_samples])
        runtime.decode_once([stream])
    stream.finish()
    runtime.decode_all([stream])

    events = [_event_to_json(ev) for ev in stream.current_events()]
    summary = {
        "wav": str(Path(args.wav).resolve()),
        "channel": args.channel,
        "mode": "baseline" if args.baseline else "deployed",
        "frames": stream.frame_offset,
        "events": events,
        "triggered_events": [_event_to_json(ev) for ev in stream.triggered_events],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fout:
            json.dump(summary, fout, ensure_ascii=False, indent=2)
            fout.write("\n")
    if args.probs_out:
        if not stream.probs:
            raise RuntimeError("no posterior frames captured")
        np.save(args.probs_out, np.concatenate(stream.probs, axis=0))


if __name__ == "__main__":
    main()
