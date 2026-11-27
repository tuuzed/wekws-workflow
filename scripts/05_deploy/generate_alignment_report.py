#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the four-way streaming alignment report from ONNX-only checks.

Four paths, one table:
1. Python ONNX posterior + upstream ctc_prefix_beam_search (baseline anchor)
2. Native posterior decoder (ctc_decoder_cli)
3. Native streaming engine (kws_host_cli)
4. WebUI streaming session (same onnx_kws_core runtime used by infer_webui.py)

Run from repository root after ``source scripts/common/env.sh``.
"""

import json
import re
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WAVS = (
    "android/tests/data/ch0.wav",
    "android/tests/data/ch_nihao.wav",
)
DEFAULT_OUTPUT = "docs/evaluation/streaming_alignment_report.md"


def _run(wav):
    command = [
        "bash", "scripts/05_deploy/compare_scores.sh", "--onnx-only", wav,
    ]
    result = subprocess.run(
        command, cwd=str(WORKSPACE_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, universal_newlines=True, check=True,
    )
    return result.stdout


def _parse_event(text):
    """Extract keyword/score/start/end from any of the four output formats."""
    if not text:
        return None
    match = re.search(
        r'"keyword":\s*"([^"]+)".*?"score":\s*([0-9.]+).*?'
        r'"start_frame":\s*(\d+).*?"end_frame":\s*(\d+)', text, re.S)
    if match:
        keyword, score, start, end = match.groups()
        return keyword, float(score), int(start), int(end)
    match = re.search(
        r"(?:detected\s+(\S+)|detected=([^\s]+))\s+([0-9.]+)\s+(\d+)\s+(\d+)",
        text)
    if match:
        keyword = match.group(1) or match.group(2)
        return keyword, float(match.group(3)), int(match.group(4)), int(match.group(5))
    match = re.search(
        r"chunk ch=\d+\s+(\S+)\s+score=([0-9.]+)\s+(\d+)-(\d+)", text)
    if match:
        return match.group(1), float(match.group(2)), int(match.group(3)), int(match.group(4))
    return None


def _section(output, start, end=None):
    lines = output.splitlines()
    begin = next((i for i, line in enumerate(lines) if line.startswith(start)), None)
    if begin is None:
        return ""
    # Keep the header line: several sections put their payload on the same
    # line as the label (e.g. ``python score: {...}``).
    if end is None:
        return "\n".join(lines[begin:])
    stop = next((i for i in range(begin + 1, len(lines))
                 if lines[i].startswith(end)), len(lines))
    return "\n".join(lines[begin:stop])


def _cell(event):
    if event is None:
        return "rejected"
    keyword, score, start, end = event
    return f"`{keyword}` {score:.6f} ({start}-{end})"


def main():
    rows = []
    for wav in DEFAULT_WAVS:
        output = _run(wav)
        python_offline = _parse_event(_section(output, "python score:",
                                               "python streaming core"))
        python_baseline = _parse_event(_section(
            output, "python streaming core (baseline):",
            "python streaming core (deployed):"))
        python_deployed = _parse_event(_section(
            output, "python streaming core (deployed):",
            "Native posterior decoder"))
        native_decoder = _parse_event(_section(
            output, "Native posterior decoder", "Native streaming engine"))
        native_engine = _parse_event(_section(output, "Native streaming engine"))
        rows.append((Path(wav).name, python_offline, native_decoder,
                     native_engine, python_deployed, python_baseline,
                     output))

    lines = [
        "# 流式 KWS 四路对齐报告",
        "",
        "> 生成方式：`bash scripts/05_deploy/generate_alignment_report.py`（先 `source scripts/common/env.sh`）。",
        "> 全部路径使用 `assets/kws.onnx`，ONNX-only 模式不依赖 `global_cmvn`。阈值来自 `keywords[].trigger_threshold`。",
        "",
        "| WAV | Python ONNX（离线 beam） | Native posterior decoder | Native streaming engine | WebUI 流式会话（deployed） | baseline（关闭触发控制） |",
        "|---|---|---|---|---|---|",
    ]
    for name, offline, decoder, engine, deployed, baseline, _ in rows:
        lines.append(
            f"| `{name}` | {_cell(offline)} | {_cell(decoder)} | "
            f"{_cell(engine)} | {_cell(deployed)} | {_cell(baseline)} |"
        )

    lines.extend([
        "",
        "## 说明",
        "",
        "- `Python ONNX / Native posterior decoder / baseline` 展示离线 score 锚点："
        "`ch0.wav ≈ 0.783478`、`ch_nihao.wav ≈ 0.224239`。",
        "- `Native streaming engine` 与 `WebUI 流式会话` 使用部署默认触发控制"
        "（`cooldown_ms=1000`、`reset_decoder_after_trigger=true`、`reset_cache_after_trigger=false`），"
        "命中后立即 reset，因此 `ch_nihao.wav` 受控事件停在 `0.088598 / 63-107`。",
        "- baseline 与 deployed 的差异来自观察时机，不是 `sqrt(∏ token_prob)` 公式或阈值变化。",
        "- Native host CLI 与 WebUI 都按 100 ms PCM 块解码，cache reset 位于相同 chunk 边界。",
        "- 同一 channel 可上报多个关键词，Engine 不做仲裁；"
        "`currentEvents()` 是当前 beam 最近命中，不是历史最高分。",
    ])

    for name, _, _, _, _, _, output in rows:
        lines.extend([
            "",
            f"## `{name}` 原始输出",
            "",
            "```text",
            output.rstrip(),
            "```",
        ])

    out_path = WORKSPACE_ROOT / DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path.relative_to(WORKSPACE_ROOT)}")
    for line in lines[:9]:
        print(line)


if __name__ == "__main__":
    main()
