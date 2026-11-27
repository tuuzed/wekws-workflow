# Proposal

## Why

Android demo 的 `Events` 面板目前只显示 `live/final`、channel、关键词、score 和起止毫秒，而 JNI `currentEventsJson()` 已经返回完整的 Native 事件对象（包含 `token_probs`）。因此移动端无法直接核对 `sqrt(∏ token_prob)` 的构成、各 token 的帧位置与概率，也不利于对齐 WebUI/Native 的事件展示口径。

## What Changes

- Android `Events` 面板改为展示 `DetectionEvent` 的全部 Native 字段：`channel`、`keyword`、`score`、`start_ms`、`end_ms`、`token_probs`。
- `token_probs` 中每个 token 展示 `token`、`token_id`、`frame`、`prob`，并保持 `score` 的显示精度与现有公式一致。
- 保留现有 live/final 标记、按 `channel/keyword/startMs/endMs` 去重刷新、录音中事件泵和停止后 flush 语义。
- 不修改 Native `CurrentEvents()`、JNI JSON 结构、score 公式、per-keyword 阈值、cooldown/reset 或部署资产；不新增阈值字段展示。
- 非破坏性变更：现有事件判定、分数和 live/final 生命周期保持不变，只扩展 Android UI 可见信息。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `streaming-kws-runtime`: 扩展 Android demo 事件呈现需求，要求界面显示 Native `currentEvents()` 返回的完整字段，尤其是 `token_probs` 的 token、token id、帧号和概率。

## Impact

- 代码：`android/app/src/main/java/com/haibingtech/kws/MainActivity.java` 的事件渲染与可能的 `DetectionEvent.java` / `TokenProb.java` 格式化辅助；JNI 和 Native 引擎不需要改动。
- 文档：更新 Android 部署架构与 WebUI/脚本工作流文档中关于 `Events` 面板字段的说明。
- 配置：继续使用 `android/app/src/main/assets/kws.onnx` 与 `config.json`，不修改阈值 `嗨小问=0.303`、`你好问问=0.039`，不新增全局 `ctc.trigger_threshold`。
- 验证：以现有 `currentEvents()` JSON 为输入检查 Android 事件渲染覆盖全部字段；不重跑离线 FRR/FAR 或 score 对齐，因为推理与阈值逻辑未变。APK 真机构建仍受 Android SDK 可用性限制。
