# Design

## Context

参见 `proposal.md` 的 Why。当前约束来自现有实现：`scripts/04_eval/infer_core.py` 只做整段 PyTorch 推理；`android/app/src/main/cpp/engine/kws_engine.cc` 已用 ONNX cache 做流式推理但只解析、未应用 cooldown/reset 字段；`MainActivity` 只在停止后拉取事件；`config.json` 部署阈值为 `嗨小问=0.303`、`你好问问=0.039`，且没有全局阈值。

## Goals / Non-Goals

**Goals:**

- WebUI 新增麦克风流式输入，并可直接选择 Android ONNX 部署后端。
- Python 流式后端与 Android Native 共用同一模型资产、阈值、cache 形状、CTC 搜索口径和 `currentEvents()` 语义。
- 让 `cooldown_ms` / `reset_decoder_after_trigger` / `reset_cache_after_trigger` 在 Native 与 Python 两端真正生效且可验证。
- Android demo 在录音过程中实时显示事件，停止后给出 final。
- 同一 WAV 的四路对齐报告可复现：Python ONNX、Native posterior decoder、Native streaming engine、WebUI 流式会话。

**Non-Goals:**

- 不修改 `avg_30.pt` 导出的 `kws.onnx`、`config.json`、`keywords.txt`、`id_to_token.json` 的契约，也不转 `.ort`。
- 不移除现有 CLI/WebUI 的整段 PyTorch 检测和调试字段。
- 不在本次实现 Android 多 AudioRecord 独立音频流；Android 范围仅限 UI 事件泵。
- 不在当前环境产出 APK；无 Android SDK 时只做 host CLI 与 Python 对齐验证。

## Decisions

**1. 流式后端以部署 ONNX 为基准，而不是 PyTorch checkpoint**

WebUI 流式后端直接读取 `android/app/src/main/assets/kws.onnx` 与 `config.json`，用 ORT 运行 `[N,T,40]` / `[N,256,105]` -> `[N,T,2599]` / `[N,256,105]`。这样运行时直接复用部署资产，避免 PyTorch 与 ONNX 两条 posterior 路径产生细微分歧。现有 PyTorch checkpoint 整段检测保留为兼容路径，不做删除。

替代方案是让流式也走 `infer_core.KwsEngine` 的 PyTorch 模型：能够支持任意 checkpoint 下拉项，但需要额外证明与 Android ONNX 的一致性，且与部署工作点解耦，故不采用。

**2. 一个共享推理会话，多个独立流状态**

Python 端参考 Native 结构：单一 ORT Session 与单 decode worker；每个浏览器会话或 channel 独立保存波形余量、fbank 上下文、cache、frame offset、beam、per-keyword 命中与 cooldown 时间戳。这样既能对齐 Android 的“多路 = 独立音频流”，又不引入多个 ORT Session 的线程与内存复杂度。

**3. Python 增量 CTC 搜索以 Native `CtcBeamSearch` 行为为准**

`wekws/wekws/bin/stream_kws_ctc.py` 只作为 chunk/cache/beam 状态参考，因为它使用全局 `threshold` 且没有当前仓库的 per-keyword 事件语义。Python 实现需保持 beam 3 / path 20 / `prob>0.05` / `abs_tol=1e-6`，分数为 `sqrt(∏ token_prob)`，并按当前 beam 顺序为每个关键词取第一个命中。top-k 并列时按概率降序、token id 升序，降低与 Native 的浮点并列差异。

**4. cooldown/reset 放在运行时，而不是 UI**

每次解码帧产生命中后，先按 per-keyword 阈值过滤并生成本帧事件；再按 channel/关键词记录触发时间，并依据配置执行 reset。cooldown 只抑制同一 channel/关键词的重复事件，不抑制其他关键词；decoder reset 清空该 channel 当前 beam；cache reset 仅在该 channel 上按配置清零。UI 只负责去重、刷新和标注 live/final，不重新实现触发逻辑。

这样可让 Android 与 WebUI 的事件次数由同一运行时规则决定；若放在 UI，会无法被 host CLI 和 FRR/FAR 流程验证。立即 reset 会让事件分数停在首次过阈值的 beam：`ch_nihao.wav` 从 baseline `0.224239 / 63-125` 变为受控 `0.088598 / 63-107`。这不是 score 公式变化，而是触发控制改变观察时机；对齐报告必须双轨展示。

**5. live/final 通过同一事件签名更新，不刷屏**

事件签名采用 `channel/keyword/startMs/endMs`。首次出现时新增一行 live 事件；后续同签名刷新分数；流结束时 flush 剩余帧并将对应行标记为 final。Android `MainActivity` 在 decode 线程节流拉取事件，通过既有 `uiHandler` 回主线程；WebUI 用 `gr.State` 保存会话状态与签名到输出行的映射。

**6. 对齐报告增加 ONNX-only 模式**

现有 `run_compare.sh` 默认还比较 PyTorch checkpoint，会因缺少 `global_cmvn` 失败。新增 ONNX-only 对齐开关，使 Python posterior、Native posterior decoder、Native streaming engine 与 WebUI 流式事件能独立跑通；完整 PyTorch 对比仍要求先执行数据准备。报告采用双轨：`ch0=0.783478`、`ch_nihao=0.224239` 是关闭触发控制的 baseline 离线 anchor；默认 cooldown/reset 下另列受控事件（`ch_nihao=0.088598 / 63-107`）和事件次数说明。

## Risks / Trade-offs

- **cooldown/reset 改变事件次数与触发节奏 →** 实施后必须重跑 ONNX/Native 对齐、FRR/FAR，并更新 `docs/evaluation/avg30_frr_far_report.md`；阈值工作点不因本次改动静默变化。
- **Python 与 Native 的 top-k 并列、浮点顺序可能不同 →** 固定并列处理规则，使用现有容差对齐，并在对齐报告中展示分数差异。
- **Gradio 麦克风 chunk 边界与浏览器实现相关 →** 每个会话显式保留 PCM 余量；停止时 flush 并输出 final，不依赖浏览器恰好对齐 chunk。
- **Android APK 无法在当前环境构建 →** 用 `kws_host_cli` 和 `ctc_decoder_cli` 验证 Native 行为；APK 构建列为需真机/SDK 的验证项。
- **`kws_engine.h` 的 “Best hit seen so far” 注释过时 →** 以 `CurrentEvents()` 当前 beam 语义为准，不在设计中沿用注释说法。
- **新增 ONNX 后端增加代码面 →** 只新增流式路径，不改 CLI `score_file` 格式和现有整段检测入口。

## Migration Plan

1. 先实现 Python ONNX 流式核心与对齐入口，保持 WebUI 现有整段检测默认可用。
2. 接入 WebUI 流式后端与 live/final 展示，再用 `ch0.wav` / `ch_nihao.wav` 验证与 Native score 锚点一致。
3. 在 Native 运行时应用 cooldown/reset，并同步 Python 运行时；用 host CLI 验证事件次数和 reset 行为。
4. 修改 Android demo 在录音中拉取事件并实时刷新，停止后保留 final。
5. 重跑 FRR/FAR 与报告更新；确认部署阈值仍为 `0.303/0.039`，不静默替换资产。
6. 若验证失败，关闭 WebUI 新后端或回退 UI 事件泵；Native cooldown/reset 可单独回退，不影响既有离线 score 路径。
