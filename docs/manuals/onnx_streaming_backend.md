# ONNX 流式 KWS 后端使用说明

> 适用入口：`scripts/04_eval/onnx_kws_core.py` 与 WebUI 的 `Android ONNX` 后端。
> 所有命令从仓库根目录执行，先 `source scripts/common/env.sh`。

## 1. 固定契约

- 模型：`android/app/src/main/assets/kws.onnx`，**直接加载 `.onnx`，不转 `.ort`**。
- 配置：`android/app/src/main/assets/config.json`；阈值来自每个 `keywords[].trigger_threshold`，不读取也不允许全局 `ctc.trigger_threshold`。
- 输入输出：`input=[N,T,40]`、`cache=[N,256,105]` -> `output=[N,T,2599]` posterior、`r_cache=[N,256,105]`。
- ONNX 输出已经是 posterior，进入 CTC 搜索前**不做第二次 softmax**。
- score 口径：`sqrt(∏ token_prob)`，beam 3 / path 20 / `prob>0.05` / `abs_tol=1e-6`；`token_probs` / `frame_topk` 只用于调试。
- `currentEvents()` 返回每个 channel/关键词“当前 beam 中最近一次观察到的命中”，不是历史最高分；同一 channel 可同时上报多个关键词，引擎不仲裁。
- 一个共享 ORT Session；每个流式会话独立保存 PCM 余量、fbank 上下文、cache、frame offset、beam、事件与 cooldown 时间戳。WebUI 的流式回调是该会话唯一的 decode worker，运行时用锁串行化 ORT 调用。

## 2. 资产与阈值自检

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh
"$PYTHON_BIN" scripts/04_eval/onnx_kws_core.py --list_keywords
```

输出应包含：

- `嗨小问`，tokens `[1302, 1462, 2494]`，`trigger_threshold=0.303`
- `你好问问`，tokens `[1027, 1401, 2494, 2494]`，`trigger_threshold=0.039`
- `cooldown_ms=1000`、`reset_decoder_after_trigger=true`、`reset_cache_after_trigger=false`

缺失或非法 `trigger_threshold` 会在初始化时直接报错；配置里出现 `ctc.trigger_threshold` 也会报错。

## 3. 单文件流式模拟

```bash
source scripts/common/env.sh
"$PYTHON_BIN" scripts/04_eval/onnx_kws_core.py \
  --wav android/tests/data/ch0.wav \
  --chunk_ms 100 \
  --summary_json work/eval/cli/ch0.stream.json
```

`--chunk_ms 100` 模拟 WebUI / Android 常见的 100 ms PCM 块；核心会逐块保留 cache 与 beam。输出 JSON 包含：

- `events`：最终 `currentEvents()`，对应 UI 的 final 展示；
- `triggered_events`：实际触发过的事件序列，用于检查 cooldown/reset；
- `mode`：`deployed`（默认）或 `baseline`。

## 4. 双轨对齐

默认部署配置会立即应用 cooldown/reset，因此事件分数可能停在首次过阈值的 beam。以 `ch_nihao.wav` 为例：

| 模式 | 关键词 | score | 起止帧 | 说明 |
|---|---|---:|---:|---|
| baseline（关闭触发控制） | 你好问问 | `0.224239` | `63-125` | 离线 score 锚点 |
| deployed（默认 cooldown/reset） | 你好问问 | `0.088598` | `63-107` | 命中后立即 reset，观察时机变化 |

`ch0.wav` 在两种模式下都约 `0.783478 / 69-129`。baseline 模式只用于对齐锚点，不用于部署：

```bash
source scripts/common/env.sh
"$PYTHON_BIN" scripts/04_eval/onnx_kws_core.py \
  --baseline --wav android/tests/data/ch_nihao.wav --chunk_ms 100
```

完整对齐入口见 `docs/manuals/script_workflow_guide.md` 的部署验证章节。

## 5. posterior 调试对比

```bash
source scripts/common/env.sh
"$PYTHON_BIN" scripts/04_eval/onnx_kws_core.py \
  --baseline \
  --wav android/tests/data/ch0.wav \
  --chunk_ms 100 \
  --probs_out work/eval/cli/ch0.stream.probs.npy
```

该 `.npy` 可与 `android/tests/onnx_reference.py` 生成的 `probs.npy` 直接比较；当前两者最大差约 `1.7e-6`（来自 PCM 量化与浮点路径），不改变阈值判定。

## 6. 状态隔离与触发控制

- WebUI 的「麦克风流式」Tab 只在 Android ONNX 后端可用；「上传检测」Tab 支持上传音频或直接录音，PyTorch 专用设置默认折叠。
- 浏览器麦克风通常按 48 kHz 回调；WebUI 在输入边界按会话状态化重采样到 16 kHz，静态检测同样按 100 ms 源采样率分块重采样。16 kHz 输入直接透传，score 公式/阈值/事件签名不变。
- 不同 WebUI 会话通过 `gr.State` 持有独立 `StreamSession`，不会共享 PCM 余量、cache 或 beam。
- cooldown 按 channel/关键词抑制重复事件；其他关键词不受影响。
- `reset_decoder_after_trigger=true` 会清空该 channel 当前 beam；`reset_cache_after_trigger=true` 才会清零该 channel 的 ONNX cache。两者都是运行时行为，不在 UI 层重复实现。
- UI 只按 `channel/keyword/startMs/endMs` 做 live/final 去重与刷新，不重新判阈值、不改 score。

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 报 `global ctc.trigger_threshold is not allowed` | 配置引入了旧全局阈值 | 删除该字段，改用每个关键词的 `trigger_threshold` |
| 报 `ONNX input 'input' must have dynamic N and T dimensions` | 用了静态 shape 的错误模型 | 使用 `assets/kws.onnx`，不要手工转换或固定 shape |
| baseline 与默认事件分数不同 | 触发后立即 reset 改变观察时机 | 按双轨报告解释；score 公式和阈值未变 |
| WebUI 流式没有 final | 未点击“结束并生成 final” | Gradio 麦克风流结束时不会自动 flush，需点击按钮 |
| 录音/上传后报采样率错误 | 输入不是 16 kHz（浏览器常为 48 kHz） | WebUI 会自动重采样；若仍报错，检查是否绕过了 WebUI 输入边界直接调用核心 |
| 浏览器麦克风流式一直无事件 | webm/opus 块被 pydub 解成 int32，原样送入会导致幅度错误 | WebUI 已按整型宽度归一到 int16 幅度；确认使用 `scripts/04_eval/infer_webui.py`，不要直接把 Gradio 原始块喂给核心 |
