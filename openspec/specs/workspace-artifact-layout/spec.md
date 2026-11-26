# workspace-artifact-layout Specification

## Purpose

定义外层项目与上游代码之间的产物归属边界，使脚本、配置、下载物、训练产物和缓存都有稳定且可移植的位置。

## Requirements

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

### Requirement: 预置训练产物归档位于 work

超过 Git LFS 单文件上限的预置训练产物归档 SHALL 位于 `work/downloads/pretrained/`，其解包产物落入 `work/exp/` 与 `work/eval/`；获取流程 MUST 不写入 `wekws/`，归档本身不进 Git。

#### Scenario: 从外部归档恢复产物

- **WHEN** 通过外层获取入口恢复预置训练产物
- **THEN** 归档位于 `work/downloads/pretrained/`，checkpoint 与评测产物分别落入 `work/exp/hi_xiaowen/ds_tcn_ctc/` 与 `work/eval/avg_30_pt/`，`wekws/` 内不新增任何文件

#### Scenario: 归档地址与校验值外置

- **WHEN** 检查版本库中的训练产物归档
- **THEN** `ds_tcn_ctc.tar.gz` 不出现在 Git 跟踪文件中，默认地址与 sha256 由 `config/pretrained/ds_tcn_ctc.env` 记录

#### Scenario: 校验失败不改写已有产物

- **WHEN** 归档或 epoch 链校验失败
- **THEN** 流程停止，已有 `work/exp/` 与 `work/eval/` 产物保持不变

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
- **THEN** 校验流程报告未满足只读约束并失败，提示将资源放回 `work/downloads/`
