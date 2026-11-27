# Tasks

## 1. Python ONNX 流式核心

- [x] 1.1 新增 `scripts/04_eval/onnx_kws_core.py`，读取 `android/app/src/main/assets/kws.onnx` 与 `config.json`，校验动态 batch、cache 形状、per-keyword `trigger_threshold` 且拒绝全局阈值；验证：脚本能加载资产并打印每个关键词 token/阈值，缺失阈值时明确失败。
- [x] 1.2 实现 PCM 分块、fbank 上下文、cache 与 ORT 增量推理，单 Session、单 decode worker；验证：把 `android/tests/data/ch0.wav` 按 100 ms 分块喂入，输出 posterior 帧与 `android/tests/onnx_reference.py` 整段输出在既定容差内一致。
- [x] 1.3 实现增量 CTC 搜索与 `currentEvents()`：beam 3 / path 20 / `prob>0.05` / `abs_tol=1e-6`，分数 `sqrt(∏ token_prob)`，按 channel/关键词取当前 beam 最近命中；验证：baseline（关闭触发控制）下 `ch0.wav` / `ch_nihao.wav` 对齐锚点 `0.783478 / 0.224239`，默认配置下 `ch_nihao.wav` 受控事件为 `0.088598 / 63-107`。
- [x] 1.4 在 Python 运行时应用 cooldown/reset：cooldown 按 channel/关键词抑制重复事件，decoder/cache reset 按 channel 生效；验证：构造或截取重复关键词片段，host/脚本输出显示 cooldown 内重复事件被抑制且 reset 后从空 beam/cache 重启。
- [x] 1.5 补充 `docs/manuals/` 下 ONNX 流式后端使用说明，明确不转 `.ort`、不二次 softmax、阈值来源与状态隔离；验证：文档命令可从仓库根目录执行且路径与 `scripts/common/env.sh` 一致。

## 2. WebUI 流式输入

- [x] 2.1 在 `scripts/04_eval/infer_webui.py` 增加后端选择器（现有 PyTorch 整段检测 / Android ONNX）与麦克风 streaming 组件，使用 `gr.State` 隔离会话；验证：启动 WebUI 后原有上传检测仍可用，后端列表包含 Android ONNX。
- [x] 2.2 将流式 PCM 块接入 ONNX 核心，展示 live/final 事件并按 `channel/keyword/startMs/endMs` 签名刷新同一行；验证：用 Gradio 客户端或浏览器喂入 `ch0.wav` 分块，停止前出现 live 事件，停止后对应行标记 final 且分数更新。
- [x] 2.3 更新 WebUI 文档中的启动命令、麦克风权限说明和 live/final 语义；验证：按文档启动 `--checkpoint avg_30.pt --port 7860` 的 WebUI 可打开并看到新控件。

## 3. Native cooldown/reset

- [x] 3.1 在 `android/app/src/main/cpp/engine/kws_engine.cc` 增加每 channel/关键词的 cooldown 时间戳，并在产生过阈值事件后按 `cooldown_ms` 抑制重复事件，不影响其他关键词；验证：`kws_host_cli` 对重复关键词片段输出显示 cooldown 内不再重复上报。
- [x] 3.2 在 Native 运行时按 `reset_decoder_after_trigger` 与 `reset_cache_after_trigger` 执行对应 channel 的 reset；验证：切换配置或临时配置分别打开/关闭 reset，`kws_host_cli` 的事件次数与后续命中行为符合设计。
- [x] 3.3 修正过时注释并更新 Native 运行时行为文档，明确以 `CurrentEvents()` 当前 beam 语义为准；验证：`kws_engine.h` 注释不再声称历史最高分，文档说明与实现一致。

## 4. Android demo 实时事件

- [x] 4.1 修改 `android/app/src/main/java/com/haibingtech/kws/MainActivity.java`，在 decode 线程节流调用事件拉取并通过既有 `uiHandler` 刷新 UI，停止后仍先 `finish` 再 final drain；验证：`git diff` 显示录音循环内事件泵且 UI 更新仍走主线程；无 SDK 时记录 APK 构建为待真机验证项。
- [x] 4.2 实现 live/final 签名映射，首次事件显示 live、同签名刷新分数、停止后标记 final，避免逐帧刷屏；验证：代码审查确认签名 key 与去重逻辑一致，host CLI 行为未受影响。
- [x] 4.3 更新 Android demo 说明，标注仅 UI 事件泵、不扩展多 AudioRecord 音频流；验证：文档中的包名仍为 `com.haibingtech.kws`，且未承诺不存在的多路能力。

## 5. 对齐与重验证

- [x] 5.1 为 `android/tests/run_compare.sh` 与 `android/tests/onnx_reference.py` 增加 ONNX-only 对齐模式，避免缺少 `global_cmvn` 时失败；验证：`bash scripts/05_deploy/compare_scores.sh --onnx-only android/tests/data/ch0.wav` 输出 Python ONNX 与 Native decoder score，baseline 锚点 `0.783478` 在容差内，并显示默认触发控制下的受控事件。
- [x] 5.2 扩展对齐报告为四路：Python ONNX、Native posterior decoder、Native streaming engine、WebUI 流式会话；验证：`ch0.wav` 与 `ch_nihao.wav` 的关键词、阈值判定和 score 同表展示，WebUI 流式会话与离线路径判定一致。
- [x] 5.3 cooldown/reset 生效后重跑 FRR/FAR 并更新 `docs/evaluation/avg30_frr_far_report.md`；验证：报告保留部署工作点 `嗨小问=0.303` / `你好问问=0.039`，新增事件次数/cooldown 影响说明，且 `awk '$2 <= 0.2 {print; exit}'` 选点流程未丢失。
- [x] 5.4 执行整体验收：`bash scripts/05_deploy/build_android_host.sh`、`bash scripts/05_deploy/compare_scores.sh --onnx-only android/tests/data/ch_nihao.wav`、`openspec validate --all --strict`，并检查 `git status --short`；验证：全部通过，`wekws/` 无新增文件，部署 assets 未被静默改写。
