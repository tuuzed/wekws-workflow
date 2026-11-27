# Proposal

## Why

现有 WebUI 只能对完整音频做一次性 PyTorch 推理，Android demo 虽然底层已具备 `AcceptWaveform`/`DecodeOnce`/`CurrentEvents` 流式能力，但 UI 只在停止后拉取事件；同时 `config.json` 中已有的 cooldown/reset 字段并未在 Native 真正生效。需要把 WebUI、Python ONNX 后端和 Android UI 统一到同一套流式 KWS 语义，并让触发控制真正可验证。

## What Changes

- 新增 Android ONNX 同口径的 Python 流式推理后端：直接加载 `android/app/src/main/assets/kws.onnx` 与 `config.json`，使用 ORT 的 `[N,T,40]`/`[N,256,105]` 输入输出和 per-keyword 阈值，不转 `.ort`。
- WebUI 保留现有上传/整段检测，新增麦克风流式输入；每个会话维护 PCM 余量、fbank 上下文、cache、增量 CTC beam 和事件状态，并展示 live/final 事件。
- Android `MainActivity` 在录音过程中周期性拉取 `currentEvents()`，实时刷新事件；停止后仍执行 `Finish()` 并输出 final。范围仅限 UI 事件泵，不扩展多 AudioRecord 音频流。
- **BREAKING**：Native 和 WebUI 同时开始应用 `cooldown_ms`、`reset_decoder_after_trigger`、`reset_cache_after_trigger`；重复关键词事件次数和触发节奏会变化，必须重新跑 ONNX/Native 对齐和 FRR/FAR 验证。
- 新增/扩展对齐报告：同一 WAV 对比 Python ONNX、Native `ctc_decoder_cli`、Native `kws_host_cli` 和新 WebUI 流式事件。

## Capabilities

### New Capabilities

- `streaming-kws-runtime`: 定义跨平台流式 KWS 运行时的统一行为，包括 ONNX cache 推理、增量 CTC 搜索、per-keyword 阈值、`currentEvents()` 语义、触发后 cooldown/reset、UI live/final 事件呈现以及三方对齐验收。

### Modified Capabilities

- 无。现有 `kws-workflow-scripts` 的 CLI/WebUI 兼容入口、`deployment-asset-export` 的资产导出契约保持不变；本次只新增流式行为。

## Impact

- 代码：`scripts/04_eval/infer_webui.py`、新增 `scripts/04_eval/onnx_kws_core.py`、新增对齐入口/脚本、`android/app/src/main/java/com/haibingtech/kws/MainActivity.java`、`android/app/src/main/cpp/engine/kws_engine.cc` 及必要的 JNI 桥接。
- 配置：继续读取 `android/app/src/main/assets/config.json` 与 `kws.onnx`；不修改部署阈值 `嗨小问=0.303`、`你好问问=0.039`，不新增全局 `ctc.trigger_threshold`。
- 依赖：Python 端使用已安装的 `onnxruntime`；Native host 对齐仍依赖 ORT headers，APK 构建需要 Android SDK（当前环境不保证可用）。
- 验证：重跑 `ch0.wav`/`ch_nihao.wav` 的 Python/ONNX/Native score 对齐，并采用双轨报告：关闭触发控制的 baseline 锚点仍为 `0.783478 / 0.224239`，默认 cooldown/reset 生效后 `ch_nihao.wav` 的受控事件为 `0.088598 / 63-107`；FRR/FAR 流程同步重跑，报告保留部署工作点并新增事件计数/触发控制影响说明。
- 不写入 `wekws/` submodule；不改归档 OpenSpec change；不静默替换 `avg_30.pt` 部署资产。
