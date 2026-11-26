# Spec Delta

## Purpose

定义外层项目提供的 KWS 工作流命令行为，让数据、训练、评估、推理和部署验证都能通过稳定入口显式传参完成。

## ADDED Requirements

### Requirement: 每个工作流阶段都有外层入口

项目 SHALL 在 `scripts/` 下提供数据准备、训练、续训、评估、通用导出、Android 导出、推理和部署验证入口；这些入口 MUST 可在不修改上游 `run_ctc.sh` 的前提下使用。

#### Scenario: 从外层入口启动训练

- **WHEN** 使用者调用外层训练入口
- **THEN** 入口显式传入 config、dict、train/dev 数据清单、model_dir 和 tensorboard_dir，且不会调用上游修改后的 `run_ctc.sh`

#### Scenario: 从外层入口执行部署验证

- **WHEN** 使用者调用部署验证入口
- **THEN** 入口使用外层 `work/` 中的 checkpoint 和上游只读代码完成 Native host 构建或 score 对齐

### Requirement: 训练与续训使用显式路径和 checkpoint

训练入口 SHALL 接受并使用显式 config、dict、数据清单、checkpoint、模型目录和 tensorboard 目录；初始微调 MUST 默认使用 `work/downloads/mobvoi_kws_transcription/23.pt`。

#### Scenario: 初始微调

- **WHEN** 未指定 checkpoint 时调用训练入口
- **THEN** 训练从 `work/downloads/mobvoi_kws_transcription/23.pt` 开始，产物写入 `work/exp/hi_xiaowen/ds_tcn_ctc/`，日志写入 `work/tensorboard/hi_xiaowen/ds_tcn_ctc/`

#### Scenario: 显式指定 checkpoint

- **WHEN** 调用训练或续训入口时传入 `--checkpoint`
- **THEN** 显式 checkpoint 覆盖默认值，且命令行中不依赖 `train.py` 不存在的 `--lr` 参数

#### Scenario: 迁移后的训练不回写上游

- **WHEN** 从 `work/exp/hi_xiaowen/ds_tcn_ctc/` 中的 checkpoint 启动训练或续训
- **THEN** model_dir 与 tensorboard_dir 仍位于 `work/`，且 `wekws/` 内不新增实验产物

### Requirement: 续训使用独立配置与输出目录

续训 SHALL 使用外层 resume 配置，其中学习率低于基础配置，并将输出写入与原始训练不同的独立模型目录。

#### Scenario: 准备续训配置

- **WHEN** 续训流程准备配置
- **THEN** `config/hi_xiaowen/ds_tcn_ctc_resume.yaml` 中的学习率低于 `ds_tcn_ctc.yaml`，且不修改基础配置

#### Scenario: 拒绝无效续训 checkpoint

- **WHEN** 续训入口被要求从 `init.pt` 恢复
- **THEN** 入口失败并拒绝启动，避免 epoch 重新计数导致日志混淆

### Requirement: 评估流程生成可复现的 FRR/FAR 结果

评估入口 SHALL 覆盖 average fallback、score、DET、逐关键词选点和报告记录；不得只迁移命令而丢失 `docs/evaluation/avg30_frr_far_evaluation.md` 中的完整流程。

#### Scenario: checkpoint 数不少于 30

- **WHEN** 可用 checkpoint 数达到 30
- **THEN** 评估入口生成 `avg_30.pt`，再产出 score 文件与 DET 结果

#### Scenario: checkpoint 数不足 30

- **WHEN** 可用 checkpoint 数少于 30
- **THEN** 评估入口生成实际 N 的 `avg_N.pt` 或直接使用 `final.pt`，并继续完成 score 与 DET

#### Scenario: score 与 DET 产物格式

- **WHEN** 评估入口完成 `score_ctc.py` 与 `compute_det_ctc.py`
- **THEN** `score.txt` 保留 3 位小数，DET 使用 `--step 0.001`，且 `stats.嗨_小_问.txt`、`stats.你_好_问_问.txt` 与 `det.png` 位于 `$RESULT_DIR/stats`

#### Scenario: 逐关键词独立选点

- **WHEN** 按 `awk '$2 <= 0.2 {print; exit}'` 读取两个 stats 文件
- **THEN** 当前复算得到的理论最低工作点为 `嗨小问=0.293`、`你好问问=0.038`，报告同时保留历史理论点 `0.297 / 0.000`；部署采用留余量工作点 `嗨小问=0.303`、`你好问问=0.039`

#### Scenario: 报告记录完整

- **WHEN** FRR/FAR 流程完成
- **THEN** `docs/evaluation/avg30_frr_far_report.md` 记录 selected epochs、test wav 数量、负样本总时长、FAR/FRR 与 `det.png` 路径

### Requirement: 评估调用使用显式且可解析的输入

评估入口 SHALL 显式传 dict 并使用 ASCII unicode escape 关键词；`score_ctc.py` 无 `--cmvn_file` 参数时 MUST 生成绝对路径评测 config 或固定可解析 cwd。

#### Scenario: 关键词与词典显式传入

- **WHEN** 评估入口调用 score 和 DET
- **THEN** `--keywords` 为 `\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee`，且两者都显式传入 `work/dict/hi_xiaowen`

#### Scenario: CMVN 路径可解析

- **WHEN** `$EXAMPLE_EXP/config.yaml` 中的 `cmvn_file` 仍是相对路径
- **THEN** 评估入口写出绝对路径评测 config 或固定可解析 cwd，禁止依赖旧 recipe 目录解析 CMVN

### Requirement: 推理能力与兼容入口保持不变

迁移后 CLI 与 WebUI SHALL 继续复用同一推理核心；`token_probs`、`frame_topk`、`avg_30.pt` 选择、`--detail_file`、`--dump_frame_topk` 和 `--score_file` 原格式 MUST 保持可用。

#### Scenario: CLI 保持调试输出

- **WHEN** 使用 CLI 并开启 `--detail_file` 或 `--dump_frame_topk`
- **THEN** 输出包含 token 概率或逐帧 top-k 调试信息，且 score 与未开启调试时一致

#### Scenario: WebUI 保持部署 checkpoint 选择

- **WHEN** 在 WebUI 中选择 `avg_30.pt` 并开启逐帧 top-k 调试
- **THEN** WebUI 使用部署 checkpoint 推理，并在界面展示调试信息

#### Scenario: 根目录 wrapper 可用

- **WHEN** 使用者仍从仓库根目录调用 `infer_cli.py` 或 `infer_webui.py`
- **THEN** wrapper 转发到外层脚本实现，且默认路径不再指向上游 recipe 的 `exp/`
