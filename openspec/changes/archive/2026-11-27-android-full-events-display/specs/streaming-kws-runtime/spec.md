# Spec Delta

## ADDED Requirements

### Requirement: Android 事件面板展示 Native 完整事件字段

Android 事件面板 SHALL 为每个可见事件展示 Native `currentEvents()` 返回的完整字段：`channel`、`keyword`、`score`、`start_ms`、`end_ms`，以及 `token_probs` 中每个 token 的 `token`、`token_id`、`frame`、`prob`。该展示 MUST 只反映运行时数据，不得改变 score、阈值过滤、cooldown/reset 或 live/final 语义。

#### Scenario: 单个命中的完整字段可见

- **WHEN** Android 运行时读取到一个包含 `token_probs` 的关键词事件
- **THEN** 事件面板同时显示该事件的 channel、关键词、score、起止毫秒，以及每个 token 的 token、token id、帧号和概率

#### Scenario: 多事件分别完整展示

- **WHEN** 同一 channel 或多个 channel 同时返回多个关键词事件
- **THEN** 每个事件都按自身数据显示完整字段，运行时不进行单一关键词仲裁，也不因字段展开而丢弃事件

#### Scenario: 刷新时完整字段同步更新

- **WHEN** 同一事件签名在后续解码中被刷新并得到新的 score 或 `token_probs`
- **THEN** 面板就地更新该行的全部字段，不新增重复行，也不改变现有 live/final 标记规则

#### Scenario: final 状态保留完整字段

- **WHEN** 用户停止录音并完成 final flush
- **THEN** 最终事件仍以完整字段展示，并保留 final 标记
