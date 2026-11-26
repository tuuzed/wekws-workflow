# Spec Delta

## Purpose

定义外层项目与上游代码之间的产物归属边界，使脚本、配置、下载物、训练产物和缓存都有稳定且可移植的位置。

## ADDED Requirements

### Requirement: 项目运行产物位于 work

项目脚本产生的下载物、data、dict、exp、tensorboard、exports、评测结果和 Python bytecode 缓存 SHALL 全部位于 `work/` 下。

#### Scenario: 迁移既有 recipe 产物

- **WHEN** 迁移 `wekws/examples/hi_xiaowen/s0/` 下已有的 `data`、`dict`、`exp`、`tensorboard` 和 `mobvoi_kws_transcription`
- **THEN** 对应产物位于 `work/` 的既定子目录，且 `wekws/` 内不再保留这些目录

#### Scenario: 运行 Python 入口

- **WHEN** 执行任意项目脚本或 Python 推理入口
- **THEN** 新的 bytecode 缓存写入 `work/pycache/`，不会在上游目录生成 `__pycache__`

### Requirement: 项目配置与上游配置分离

项目使用的 hi_xiaowen 配置 SHALL 位于外层 `config/hi_xiaowen/`，不得依赖修改上游 recipe 中的 yaml 来生效。

#### Scenario: 使用 `batch_size=64` 配置

- **WHEN** 训练或续训读取外层 `config/hi_xiaowen/ds_tcn_ctc.yaml`
- **THEN** 生效配置包含 `batch_size: 64`，且上游 `wekws/examples/hi_xiaowen/s0/conf/ds_tcn_ctc.yaml` 保持原样

#### Scenario: 续训使用单独配置

- **WHEN** 续训流程读取 `config/hi_xiaowen/ds_tcn_ctc_resume.yaml`
- **THEN** 续训配置与基础配置分离，且降低后的学习率只存在于续训配置中

### Requirement: 迁移后的数据清单指向 work

迁移后的 `data.list` SHALL 重新生成，其中所有 wav 路径 MUST 指向 `work/` 下的迁移后音频。

#### Scenario: 校验迁移后的 data.list

- **WHEN** 检查 `work/data/hi_xiaowen/{train,dev,test}/data.list`
- **THEN** 每条 wav 路径都位于 `work/` 内，且不包含指向 `wekws/examples/hi_xiaowen/s0/data/` 的旧绝对路径

#### Scenario: 拒绝直接复用旧清单

- **WHEN** 准备脚本发现待使用的 `data.list` 仍含旧上游路径
- **THEN** 准备流程失败并提示必须重新生成 `data.list`

### Requirement: 既有训练产物平稳迁移

迁移 `exp/` 与 `tensorboard/` 时 SHALL 先建立基线清单并检测写入进程；迁移 MUST 保持 checkpoint 链、config、`avg_30.pt` 和日志可解析，且在验收前可回退。

#### Scenario: 检测到写入进程

- **WHEN** 迁移脚本发现 `train.py` 或 `torchrun` 正在写上游 `exp/`
- **THEN** 不直接移动在用文件，先复制到 `work/` 并校验，待写入结束后原子切换且保留旧目录

#### Scenario: 无写入进程

- **WHEN** 迁移脚本确认无写入进程
- **THEN** 使用同文件系统 `mv` 原子迁移，并对比文件数、大小、sha256 与 epoch 连续性

#### Scenario: checkpoint 链保持完整

- **WHEN** 迁移完成
- **THEN** `work/exp/hi_xiaowen/ds_tcn_ctc/` 保留全部 epoch `.pt`/`.yaml`、`config.yaml` 与 `avg_30.pt`，tensorboard 日志可读，且 `avg_30.pt` 基线不变

#### Scenario: 校验失败可回退

- **WHEN** 迁移后校验发现文件数、sha256 或 epoch 链不一致
- **THEN** 从保留的旧目录恢复，并停止后续清理步骤

### Requirement: 评测中间产物与报告有固定归属

FRR/FAR 评测产物 SHALL 分为两类：`score.txt`、`stats/*.txt` 与 `det.png` 位于 `work/eval/`，最终选点报告位于 `docs/evaluation/`，不得写回上游 recipe。

#### Scenario: 评测中间产物落点

- **WHEN** `eval_ctc.sh` 完成 score 与 DET
- **THEN** `score.txt`、`stats.嗨_小_问.txt`、`stats.你_好_问_问.txt` 与 `det.png` 位于 `work/eval/` 下对应结果目录

#### Scenario: 评测报告落点

- **WHEN** FRR/FAR 选点完成
- **THEN** 报告写入 `docs/evaluation/avg30_frr_far_report.md`，且引用 `work/eval/` 下的 stats 与 `det.png` 路径

### Requirement: 上游目录免受项目写入

项目工作流 SHALL 不在 `wekws/` 内创建或写入任何项目落地文件。

#### Scenario: 完整工作流后的只读校验

- **WHEN** 完成数据准备、训练、评估、导出和推理后检查 `wekws/`
- **THEN** `wekws/` 中不存在 `data`、`dict`、`exp`、`tensorboard`、`mobvoi_kws_transcription` 或 `__pycache__`

#### Scenario: 下载资源位置错误

- **WHEN** `mobvoi_kws_transcription` 仍位于 `wekws/examples/hi_xiaowen/s0/`
- **THEN** 校验流程报告未满足只读约束，迁移不能标记为完成
