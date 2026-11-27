# streaming-kws-runtime Specification

## Purpose

定义跨平台流式 KWS 的可观察行为：WebUI、Python ONNX 后端和 Android 客户端必须对同一音频流产生同一套阈值过滤后的关键词事件，并在触发后一致地应用 cooldown 与 reset 控制。

## Requirements

### Requirement: 流式会话按块消费 PCM 并保留推理状态

流式运行时 SHALL 接受连续 PCM 块，为每个会话保留波形余量、特征上下文和模型 cache；每块可用帧 MUST 只经过一次增量推理，且不得为每一块重新加载部署模型。

#### Scenario: 连续 PCM 块产生连续 posterior 帧

- **WHEN** 客户端以 16 kHz 单声道 PCM 块持续喂入同一流式会话
- **THEN** 运行时逐块产出 posterior 帧并保留跨块 cache，后续帧与整段输入顺序一致

#### Scenario: 会话之间状态隔离

- **WHEN** 两个浏览器会话或 channel 同时流式输入
- **THEN** 每个会话的波形余量、特征上下文、cache 与 frame offset 互不影响

### Requirement: 流式推理直接使用部署 ONNX 资产

Python 流式后端 SHALL 直接加载部署的 `kws.onnx` 与 `config.json`，使用动态 batch 与 cache 输入，输出 posterior；MUST NOT 转换 `.ort`，MUST NOT 对 ONNX 输出再次 softmax。

#### Scenario: 加载部署模型

- **WHEN** 流式后端启动并选择 Android ONNX 后端
- **THEN** 它使用部署 assets 中的 `kws.onnx` 和 `config.json`，输入形状为 `[N,T,40]` 与 `[N,256,105]`，输出为 `[N,T,2599]` posterior 与 `[N,256,105]` cache

#### Scenario: posterior 不再二次 softmax

- **WHEN** ONNX 输出进入 CTC 搜索
- **THEN** 搜索直接使用 posterior，事件分数与部署端一致

### Requirement: 增量 CTC 搜索保持离线 score 口径

流式运行时 SHALL 在当前 beam 上按 `sqrt(∏ token_prob)` 计算关键词分数，且 beam 3、path 20、`prob>0.05`、`abs_tol=1e-6` 的口径 MUST 与离线 score 一致。

#### Scenario: 命中关键词的分数计算

- **WHEN** 某关键词 token 序列出现在当前 beam 中
- **THEN** 事件分数等于这些 token 节点概率乘积的平方根，起止帧对应第一个和最后一个 token 节点

#### Scenario: 多关键词同时命中

- **WHEN** 同一会话当前 beam 中同时包含两个已配置关键词
- **THEN** 两个关键词都会被报告，运行时不做单一关键词仲裁

### Requirement: 事件按关键词独立阈值过滤

运行时 SHALL 使用部署配置中每个关键词的 `trigger_threshold` 过滤事件；MUST NOT 使用全局 `ctc.trigger_threshold`，缺失或非法阈值 MUST 在_runtime 初始化_时失败。

#### Scenario: 缺失或非法阈值

- **WHEN** 部署配置中某个关键词缺少 `trigger_threshold` 或其值非法
- **THEN** 运行时初始化失败并指出具体关键词，不进入推理状态

#### Scenario: 独立阈值通过

- **WHEN** 两个关键词的当前 beam 分数分别达到各自阈值
- **THEN** 两个关键词都产生事件；未达到自身阈值的关键词不产生事件

### Requirement: currentEvents 语义在两端一致

运行时 SHALL 为每个 channel/关键词暴露“当前 beam 中最近一次观察到的命中”，而不是历史最高分；命中在未被替换或重置前 MUST 保持可见。

#### Scenario: 历史高分消失

- **WHEN** 某关键词早期高分从当前 beam 消失，而当前 beam 中仍有较低分命中
- **THEN** 对外报告当前较低分命中，不报告历史最高分

#### Scenario: 重置后清除命中

- **WHEN** 会话或 channel 被显式重置
- **THEN** 该会话/channel 的所有关键词命中状态被清除，直到新的命中出现

### Requirement: 触发后 cooldown 与 reset 生效

运行时 SHALL 在事件通过后按部署配置应用 `cooldown_ms`、`reset_decoder_after_trigger` 与 `reset_cache_after_trigger`；cooldown 按 channel/关键词生效，decoder/cache reset 按 channel 生效。

#### Scenario: cooldown 抑制重复事件

- **WHEN** 某 channel 的关键词触发后仍处于该关键词的 cooldown 窗口
- **THEN** 同一 channel/关键词的后续重复事件被抑制，其他关键词仍可触发

#### Scenario: decoder reset 清空当前 beam

- **WHEN** `reset_decoder_after_trigger` 为真且某 channel 触发事件
- **THEN** 该 channel 的当前 beam 被清空，后续检测从空 beam 重新开始

#### Scenario: cache reset 按配置执行

- **WHEN** `reset_cache_after_trigger` 为真或为假且某 channel 触发事件
- **THEN** 为真时该 channel 的模型 cache 被清零，为假时 cache 被保留

### Requirement: WebUI 提供流式输入和 live/final 事件

WebUI SHALL 保留现有整段检测能力，并提供麦克风流式输入；流式过程中显示 live 事件，流结束后显示 final 事件，且不同浏览器会话状态隔离。

#### Scenario: 录音中显示 live 事件

- **WHEN** 用户开始麦克风流式输入且关键词在录音过程中达到阈值
- **THEN** 事件在停止录音前显示，并标注为 live 状态

#### Scenario: 停止后显示 final 事件

- **WHEN** 用户停止流式输入
- **THEN** 运行时 flush 剩余帧并显示 final 事件，final 分数替换或补充对应 live 事件

### Requirement: Android demo 在录音中实时刷新事件

Android demo SHALL 在录音进行中周期性读取并显示已过阈值的事件；停止后 MUST 先 flush 再显示 final 事件。本需求不要求同时喂入多个独立 AudioRecord 音频流。

#### Scenario: 录音中实时显示

- **WHEN** Android demo 正在录音且某 channel/关键词达到阈值
- **THEN** 对应事件在用户停止录音前出现在界面上

#### Scenario: 停止后 final 显示

- **WHEN** 用户停止录音
- **THEN** demo 先完成剩余帧 flush，再显示最终事件与最终分数

### Requirement: 对齐报告覆盖 Python、Native 与 WebUI 流式路径

验证入口 SHALL 对同一 WAV 输出 Python ONNX、Native posterior decoder、Native streaming engine 与 WebUI 流式会话的对比结果，包含关键词、分数、起止帧和阈值判定。

#### Scenario: 固定样本 baseline 对齐

- **WHEN** 对 `ch0.wav` 与 `ch_nihao.wav` 在关闭 cooldown/reset 的 baseline 模式下执行对齐
- **THEN** 报告显示 Python 与 Native 的离线 score 锚点 `0.783478` 与 `0.224239` 在容差内一致，且关键词与阈值判定一致

#### Scenario: 默认触发控制后的事件分数

- **WHEN** 对 `ch_nihao.wav` 使用部署配置中默认生效的 cooldown/reset 执行对齐
- **THEN** 报告同时列出 baseline 锚点 `0.224239 / 63-125` 与受控事件 `0.088598 / 63-107`，并说明分数变化来自命中后立即 reset，而非 score 口径变化

#### Scenario: cooldown/reset 后重新验证

- **WHEN** cooldown/reset 行为生效后执行对齐
- **THEN** 对齐报告包含事件次数变化说明，FRR/FAR 流程被重新执行且报告工作点与事件计数被更新
