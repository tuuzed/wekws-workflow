# 流式 KWS 四路对齐报告

> 生成方式：`bash scripts/05_deploy/generate_alignment_report.py`（先 `source scripts/common/env.sh`）。
> 全部路径使用 `assets/kws.onnx`，ONNX-only 模式不依赖 `global_cmvn`。阈值来自 `keywords[].trigger_threshold`。

| WAV | Python ONNX（离线 beam） | Native posterior decoder | Native streaming engine | WebUI 流式会话（deployed） | baseline（关闭触发控制） |
|---|---|---|---|---|---|
| `ch0.wav` | `嗨小问` 0.783478 (69-129) | `嗨小问` 0.783478 (69-129) | `嗨小问` 0.783477 (69-129) | `嗨小问` 0.783477 (69-129) | `嗨小问` 0.783477 (69-129) |
| `ch_nihao.wav` | `你好问问` 0.224239 (63-125) | `你好问问` 0.224239 (63-125) | `你好问问` 0.088598 (63-107) | `你好问问` 0.088597 (63-107) | `你好问问` 0.224240 (63-125) |

## 说明

- `Python ONNX / Native posterior decoder / baseline` 展示离线 score 锚点：`ch0.wav ≈ 0.783478`、`ch_nihao.wav ≈ 0.224239`。
- `Native streaming engine` 与 `WebUI 流式会话` 使用部署默认触发控制（`cooldown_ms=1000`、`reset_decoder_after_trigger=true`、`reset_cache_after_trigger=false`），命中后立即 reset，因此 `ch_nihao.wav` 受控事件停在 `0.088598 / 63-107`。
- baseline 与 deployed 的差异来自观察时机，不是 `sqrt(∏ token_prob)` 公式或阈值变化。
- Native host CLI 与 WebUI 都按 100 ms PCM 块解码，cache reset 位于相同 chunk 边界。
- 同一 channel 可上报多个关键词，Engine 不做仲裁；`currentEvents()` 是当前 beam 最近命中，不是历史最高分。

## `ch0.wav` 原始输出

```text
onnx probs shape: (200, 2599)
python score: {"keyword": "嗨小问", "score": 0.783478, "start_frame": 69, "end_frame": 129}
python streaming core (baseline): [{"keyword": "嗨小问", "score": 0.783477, "start_frame": 69, "end_frame": 129}]
python streaming core (deployed): [{"keyword": "嗨小问", "score": 0.783477, "start_frame": 69, "end_frame": 129}]
Native posterior decoder on ONNX posteriors:
detected 嗨小问 0.783478 69 129
Native streaming engine (fbank -> ONNX -> CTC):
chunk ch=0 嗨小问 score=0.783477 69-129
chunk ch=0 嗨小问 score=0.783477 69-129
chunk ch=0 嗨小问 score=0.783477 69-129
chunk ch=0 嗨小问 score=0.783477 69-129
chunk ch=0 嗨小问 score=0.783477 69-129
chunk ch=0 嗨小问 score=0.783477 69-129
chunk ch=0 嗨小问 score=0.783477 69-129
chunk ch=0 嗨小问 score=0.783477 69-129
channel=0 detected=嗨小问 score=0.783477 start_ms=690 end_ms=1290 token_probs=[嗨=0.986322@69,小=0.997568@109,问=0.623866@129]
```

## `ch_nihao.wav` 原始输出

```text
onnx probs shape: (191, 2599)
python score: {"keyword": "你好问问", "score": 0.224239, "start_frame": 63, "end_frame": 125}
python streaming core (baseline): [{"keyword": "你好问问", "score": 0.22424, "start_frame": 63, "end_frame": 125}]
python streaming core (deployed): [{"keyword": "你好问问", "score": 0.088597, "start_frame": 63, "end_frame": 107}]
Native posterior decoder on ONNX posteriors:
detected 你好问问 0.224239 63 125
Native streaming engine (fbank -> ONNX -> CTC):
chunk ch=0 你好问问 score=0.0885978 63-107
chunk ch=0 你好问问 score=0.0885978 63-107
chunk ch=0 你好问问 score=0.0885978 63-107
chunk ch=0 你好问问 score=0.0885978 63-107
chunk ch=0 你好问问 score=0.0885978 63-107
chunk ch=0 你好问问 score=0.0885978 63-107
chunk ch=0 你好问问 score=0.0885978 63-107
chunk ch=0 你好问问 score=0.0885978 63-107
chunk ch=0 你好问问 score=0.0885978 63-107
chunk ch=0 你好问问 score=0.0885978 63-107
channel=0 detected=你好问问 score=0.088598 start_ms=630 end_ms=1070 token_probs=[你=0.953913@63,好=0.989329@85,问=0.120313@96,问=0.069133@107]
```
