# Design

## Context

Native `CurrentEvents()` 与 JNI `currentEventsJson()` 已经输出事件对象及 `token_probs`；Java 层 `DetectionEvent` / `TokenProb` 也已解析并持有这些字段。当前只有 `MainActivity.renderEvents()` 把事件压成一行摘要，且 `eventView` 是非滚动 `TextView`。环境约束见 `proposal.md`：不修改 Native/JNI 输出、score 公式、阈值、cooldown/reset、部署资产或 AudioRecord 流数量。

## Goals / Non-Goals

**Goals:**

- 把 Native 事件对象中的每个字段原样呈现在 Android `Events` 面板中，包括逐 token 的 `token`、`token_id`、`frame`、`prob`。
- 保持现有事件签名、去重、就地刷新、live/final 标记和停止后 flush 流程不变。
- 让展开后的事件列表在屏幕上可读、可滚动，不挤掉控制按钮和日志区。
- 用确定性数字格式展示浮点值，便于与 WebUI/Native 报告对照。

**Non-Goals:**

- 不新增阈值字段；Native 事件 JSON 本身不包含 `trigger_threshold`。
- 不改 `CurrentEvents()` 语义、JNI JSON schema、CTC 搜索、score、阈值过滤或触发控制。
- 不把原始 JSON 字符串直接倾倒到 UI，也不新增独立的事件队列或多路音频采集。
- 不重新选部署工作点、不重跑离线 FRR/FAR 或三方 score 对齐。

## Decisions

1. **以现有 Java 数据模型作为唯一展示输入。** 在 `DetectionEvent` / `TokenProb` 上增加面向事件面板的格式化辅助，由 `MainActivity` 拼接 live/final 前缀；不修改 JNI 或 Native。  
   *理由*：字段已完整解析，改动最小且不会触碰推理口径。*替代方案*：直接展示原始 JSON——可读性差且无法自然保留 live/final 与去重行；改 JNI schema——无收益且增加跨平台风险。

2. **每个事件使用多行块展示。** 首行保留 `live/final`、channel、关键词、score、起止毫秒；随后每个 `token_probs` 项单独一行，显示 `token`、`token_id`、`frame`、`prob`。浮点统一 `Locale.US`、6 位小数，与当前 score 显示精度一致。  
   *理由*：完整信息仍可逐项扫读，且不会被长 token 列表挤成不可读的单行。*替代方案*：单行追加全部 token——在 4 token 关键词和小屏设备上会溢出。

3. **刷新规则完全沿用现有 map。** key 仍为 `channel/keyword/startMs/endMs`；同一事件被刷新时替换整个 `DetectionEvent`（含新的 `token_probs`）并保留原 live/final 状态，final flush 时统一标记 final。  
   *理由*：满足“就地更新、不新增重复行”，同时避免把展示层状态变成事件语义的一部分。*替代方案*：按内容变化新增行——会重复上报同一事件，破坏现有对齐口径。

4. **为事件区增加独立滚动容器。** 将 `eventView` 放入带权重的 `ScrollView`，日志区继续保持自己的滚动容器。  
   *理由*：`token_probs` 展开后高度不可预测，必须防止按钮和日志被挤出屏幕。*替代方案*：继续 `wrap_content`——在小屏或多个事件同时出现时不可用。

5. **验证聚焦字段覆盖而非推理重算。** 以现有 `currentEvents()` JSON 样例检查格式化输出是否覆盖全部字段；若 Android SDK 可用则构建 APK，否则明确记录无法真机验证。  
   *理由*：本变更只改变展示，不改变任何事件判定或分数计算。*替代方案*：重跑 FRR/FAR 和 score 对齐——成本高且对本变更无判别力。

## Risks / Trade-offs

- [多行事件增加 UI 高度和刷新文本量] → 用独立 `ScrollView` 限制区域，并继续按 100 ms 事件泵就地替换文本，不逐帧追加日志。
- [格式化改动意外影响其他调用方] → 使用专门的事件面板格式化方法，不复用 `toString()` 作为通用协议输出。
- [无法在 Android SDK 缺失环境做真机验收] → 至少完成字段覆盖的静态/样例验证，并在任务中明确标注 APK 构建与真机验证为条件步骤。
- [误把阈值或 cooldown 信息加入面板] → 严格按 Native `currentEvents()` JSON 字段展示，不读取配置补充字段。

## Migration Plan

无需数据或配置迁移；升级 Android demo 即可获得完整字段展示。回退方式为撤销本次 UI/文档改动，Native、JNI、模型和阈值保持原样。
