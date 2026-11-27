# Tasks

## 1. 事件数据展示格式化

- [x] 1.1 在 `TokenProb` 增加面向事件面板的格式化方法，输出 `token`、`token_id`、`frame`、`prob`（`Locale.US`、6 位小数）；验证：样例 `TokenProb` 格式化字符串包含四个字段，且概率为 6 位小数。
- [x] 1.2 在 `DetectionEvent` 增加面向事件面板的多行格式化方法，输出 `channel`、`keyword`、`score`、`start_ms`、`end_ms` 和全部 `token_probs` 行；验证：用现有 `currentEvents()` JSON 样例解析后格式化，所有字段均出现，且 token 行数等于 `token_probs` 数量。

## 2. Android UI 接入

- [x] 2.1 修改 `MainActivity.renderEvents()` 使用新的格式化方法生成多行事件块，并保持 `live/final` 前缀、事件签名去重、就地刷新和 final 标记逻辑不变；验证：同一事件签名刷新后界面事件块数量不增加，字段值随刷新更新。
- [x] 2.2 在 `activity_main.xml` 中给 `eventView` 增加独立 `ScrollView` 与合理权重，日志区仍保持可滚动；验证：展开多个事件或 4-token 关键词后，Start/Stop/Debug 按钮和日志区仍可见、可操作。

## 3. 文档与验证

- [x] 3.1 更新 `docs/architecture/android_deployment_architecture.md` 与 `docs/manuals/script_workflow_guide.md`，说明 Android Events 面板显示 Native 完整字段及 `token_probs` 子字段；验证：文档字段清单与 delta spec 一致，且未声称新增阈值或 cooldown 字段。
- [x] 3.2 用 `currentEvents()` JSON 样例核对面板输出覆盖全部字段，并确认 Native/JNI/模型/阈值文件未改动；验证：`git diff --stat` 仅包含 Java、布局、文档与 OpenSpec change 文件，`openspec validate --all --strict` 通过。
- [x] 3.3 条件验证：若 Android SDK 可用，运行 `bash scripts/05_deploy/build_apk.sh` 并真机确认多事件、多 token 展示与滚动；验证：SDK 可用时构建成功且真机显示完整字段；SDK 不可用时在交付说明中记录未执行原因。
