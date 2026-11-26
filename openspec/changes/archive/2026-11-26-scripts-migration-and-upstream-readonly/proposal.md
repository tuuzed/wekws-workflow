# Proposal

## Why

当前项目脚本、配置、下载物、dict、数据和实验产物仍分散在只读上游 `wekws/examples/hi_xiaowen/s0/`，且 `wekws/` 仍是带本地修改的普通目录；这让上游无法保持干净、路径不可移植，训练与部署也依赖易变的相对路径。外层仓库虽已完成首次提交并提交部署用 `kws.onnx`，但尚未把入口脚本、产物目录和上游依赖收口到稳定、可复现的项目结构中。FRR/FAR 评测虽已完成，但其选点流程、统计产物和 CMVN/cwd 约束尚未成为外层脚本与 OpenSpec 的可执行契约，迁移时容易丢失。

## What Changes

- 新增外层 `scripts/`、`config/`、`work/` 与 `scripts/common/env.sh`，统一所有路径、Python 环境与 Python bytecode 落点。
- 将下载物、`data/`、`dict/`、`exp/`、`tensorboard/` 从上游 recipe 迁到 `work/`，并重新生成 `data.list`，使其 wav 路径全部指向 `work/`。
- 新增数据准备、训练/续训、评估、通用 ONNX/JIT 导出、Android ONNX 导出、推理和部署验证脚本，全部显式传 config/dict/checkpoint/model_dir。
- 修正 `infer_core.py` / CLI / WebUI 的默认路径，根目录保留兼容 wrapper；保留 `token_probs`、`frame_topk`、`--score_file` 原格式等现有能力。
- 修正 Android 导出与 host 对齐脚本默认值：部署固定使用 `avg_30.pt`，`--keyword_thresholds` 必填，写入 `嗨小问=0.303`、`你好问问=0.039`，且不写全局 `ctc.trigger_threshold`。
- 完整保留 FRR/FAR 评测与选点流程：`score.txt` 保留 3 位小数，DET 使用 `--step 0.001`，生成 `stats/*.txt` 与 `det.png`，按 `awk '$2 <= 0.2 {print; exit}'` 逐关键词独立选点，并记录到 `docs/evaluation/avg30_frr_far_report.md`。
- 评测入口 MUST 显式传 `--dict` 且使用 ASCII unicode escape 关键词；`score_ctc.py` 无 `--cmvn_file` 参数时必须生成绝对路径评测 config 或固定可解析 cwd，不得依赖旧 recipe 目录。
- 现有训练产物平稳迁移：迁移前记录文件数、大小、sha256、epoch 连续性与 `avg_30.pt` 基线；检测无写入进程时同文件系统原子移动，有写入进程时先复制校验再切换，验收前保留旧目录；不得丢失 checkpoint 链、config 或 tensorboard 日志，也不得让部署默认 checkpoint 漂移到更新 epoch。
- 恢复 `wekws/` 中的两处本地修改，将 `wekws` 转为 pin 到 `6a45aeb994dd81c0969ff877a5a7c46d60ed0c86` 的 git submodule；其后上游目录只读、干净。
- 更新文档中的推荐命令与路径，并以可执行脚本和 assets 为事实来源清理旧 `15.pt` / 联调描述。
- **BREAKING**：依赖 `wekws/examples/hi_xiaowen/s0/exp/...` 或上游 `run_ctc.sh` 的旧调用方式不再受支持；迁移后必须通过外层脚本和 `work/` 路径调用。

## Capabilities

### New Capabilities

- `workspace-artifact-layout`: 定义项目脚本、配置、运行产物和缓存的归属目录，保证所有落地产物位于 `work/`、既有训练产物平稳迁移且上游目录不被写入。
- `kws-workflow-scripts`: 定义外层提供的训练、续训、评估、推理与部署验证命令契约，包括显式路径、checkpoint 选择、FRR/FAR 选点流程、既有 checkpoint 链保持和现有调试能力保持。
- `deployment-asset-export`: 定义通用 ONNX/JIT 与 Android 动态 batch ONNX 资产导出行为，包括固定部署 checkpoint、FRR/FAR 保守工作点的 per-keyword 阈值和 posterior 输出约定。
- `upstream-submodule-isolation`: 定义 `wekws` 作为只读、干净、pin 住 commit 的 git submodule 的约束与恢复流程。

### Modified Capabilities

无。当前 OpenSpec 项目中没有已有 spec。

## Impact

- 新增：`scripts/**`、`config/hi_xiaowen/**`、`work/**`（gitignore）、`.gitmodules`。
- 修改：`infer_core.py`、`infer_cli.py`、`infer_webui.py`、`android/export/export_ctc_onnx.py`、`android/tests/run_compare.sh`、相关文档。
- 新增评测记录：`docs/evaluation/avg30_frr_far_report.md`，保留 selected epochs、test wav 数量、负样本总时长、`stats/*.txt` 与 `det.png` 路径。
- 依赖：`wekws` submodule pin、conda `wekws` 环境、`torch/torchaudio==2.5.1`、`onnxruntime==1.23.2`、conda-forge `sox`。
- 系统：训练、FRR/FAR 评估、Python 推理、Android Native host 对齐和 APK 构建入口。
