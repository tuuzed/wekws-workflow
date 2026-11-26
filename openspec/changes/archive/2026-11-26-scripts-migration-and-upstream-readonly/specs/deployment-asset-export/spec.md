# Spec Delta

## Purpose

定义 KWS 模型导出的对外资产契约，确保通用导出与 Android 部署导出都能从固定工作点生成一致、可验证的模型和配置。

## ADDED Requirements

### Requirement: 部署导出固定使用 avg_30.pt

Android 部署资产 SHALL 默认由 `work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt` 导出，并允许通过 `--checkpoint` 显式覆盖；后续新 checkpoint MUST 不自动替换部署资产。

#### Scenario: 默认导出部署资产

- **WHEN** 未指定 checkpoint 时执行 Android 导出
- **THEN** 导出使用 `avg_30.pt`，并写入 `android/app/src/main/assets/` 中的全部部署资产

#### Scenario: 显式覆盖 checkpoint

- **WHEN** 导出时传入 `--checkpoint`
- **THEN** 显式路径覆盖默认部署 checkpoint，且默认资产不会被隐式改写

#### Scenario: 新 checkpoint 不自动替换部署资产

- **WHEN** 训练产出比 `avg_30.pt` 更新的 checkpoint
- **THEN** Android 部署资产仍由 `avg_30.pt` 导出，除非显式指定新 checkpoint 并重新执行完整导出与对齐流程

### Requirement: 通用 ONNX 与 JIT 链路同时保留

项目 SHALL 保留上游通用 ONNX/JIT 导出链路；ONNX 入口 MUST 使用 `--onnx_model`，JIT 入口 MUST 使用 `--jit_model`。

#### Scenario: 导出通用 ONNX

- **WHEN** 调用通用 ONNX 导出入口
- **THEN** 使用 `--onnx_model` 在 `work/exports/hi_xiaowen/` 下生成通用 ONNX 文件

#### Scenario: 导出 JIT 模型

- **WHEN** 调用通用 JIT 导出入口
- **THEN** 使用 `--jit_model` 在 `work/exports/hi_xiaowen/` 下生成 JIT 模型文件

### Requirement: Android 导出要求显式 per-keyword 阈值

Android 导出 SHALL 要求 `--keyword_thresholds` 必填，并把每个阈值写入对应关键词的 `trigger_threshold`；runtime config MUST 不含全局 `ctc.trigger_threshold`。

#### Scenario: 缺少阈值参数

- **WHEN** 执行 Android 导出但未传 `--keyword_thresholds`
- **THEN** 导出失败，并提示参数必填及正确示例

#### Scenario: 使用最终工作点导出

- **WHEN** 导出时传入 `嗨小问=0.303,你好问问=0.039`
- **THEN** `config.json` 中两个关键词分别写入 `0.303` 与 `0.039`，且不存在全局 `ctc.trigger_threshold`

#### Scenario: 理论最低点仅用于评测记录

- **WHEN** 评测报告记录历史理论点 `嗨小问=0.297`、`你好问问=0.000`，且当前 `score.txt` 按 `awk '$2 <= 0.2 {print; exit}'` 复算得到 `0.293 / 0.038`
- **THEN** 两类选点都只写入评测报告，不直接用于部署导出；部署仍使用留余量工作点 `0.303 / 0.039`

#### Scenario: 阈值关键词不匹配

- **WHEN** `--keyword_thresholds` 包含未配置或缺少配置关键词
- **THEN** 导出失败并列出未知或缺失的关键词

### Requirement: Android ONNX 保持动态 batch 与 posterior 输出

Android 导出的 ONNX SHALL 保持动态 batch 维度和 CTC cache 形状，并输出 softmax posterior；Android Native MUST 不再进行第二次 softmax。

#### Scenario: 动态 batch 推理

- **WHEN** 以 batch `N`、帧数 `T` 运行导出的 `kws.onnx`
- **THEN** 输入形状为 `[N,T,40]` 与 `[N,256,105]`，输出形状为 `[N,T,2599]`，并返回 `[N,256,105]` cache

#### Scenario: posterior 一致性

- **WHEN** 比较 PyTorch 与 ONNX Runtime 对同一音频的输出
- **THEN** ONNX 输出直接作为 posterior 参与 CTC beam search，且不会再次经过 softmax

### Requirement: 导出资产成组一致且可验证

一次 Android 导出 SHALL 生成 `kws.onnx`、`config.json`、`keywords.txt` 与 `id_to_token.json`，并使关键词 token 映射与阈值配置一致。

#### Scenario: 关键词 token 映射校验

- **WHEN** 导出完成后读取 `id_to_token.json` 与 `config.json`
- **THEN** `嗨小问` 映射为 `[1302,1462,2494]`，`你好问问` 映射为 `[1027,1401,2494,2494]`，且 config 中 token 与之相同

#### Scenario: 部署 score 对齐

- **WHEN** 使用 `avg_30.pt` 导出资产并执行 Python/ONNX/Native score 对齐
- **THEN** 三方 score 在既定容差内一致，且检测事件与迁移前一致
