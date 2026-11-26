# Design

## Context

参见 `proposal.md` 的 Why。当前外层仓库已有首次提交 `81c82f2`，部署用 `kws.onnx` 已通过 `.gitignore` 例外入库；`android/app/src/main/assets/config.json` 已包含 `0.303 / 0.039` 的 per-keyword 阈值，Android 导出脚本也已要求 `--keyword_thresholds`。尚未落地的部分是 `scripts/`、`config/`、`work/`、`env.sh`，以及 `infer_core.py` 仍指向 `wekws/examples/hi_xiaowen/s0/exp`。`wekws/` 仍有两处修改和未跟踪的 `mobvoi_kws_transcription/`，因此还不能干净地转为 submodule。计划文档 `docs/plans/scripts_migration_and_upstream_readonly_plan.md` 已同步当前仓库状态、FRR/FAR 完整流程和 21 项 Phase 4 验收门禁，OpenSpec change 也与其对齐；实施时以可执行脚本、assets 和该计划为准。

## Goals / Non-Goals

**Goals:**

- 建立单一外层路径来源，使训练、续训、评估、导出、推理和部署验证不再依赖上游 recipe 的相对路径。
- 把全部运行产物收拢到 `work/`，并使 `wekws/` 成为干净、只读、pin 住的 submodule。
- 在迁移过程中保持 score 公式、beam 参数、per-keyword 阈值、`token_probs` / `frame_topk` 调试语义和 `score_file` 格式不变。
- 完整保留 FRR/FAR 选点流程与报告产物，保证 `score.txt`、`stats/*.txt`、`det.png` 和 `0.303 / 0.039` 保守工作点可复现。
- 提供可执行的迁移后验收门禁，覆盖路径、token id、动态 batch ONNX、三方 score 对齐、CMVN/cwd 约束和上游只读性。

**Non-Goals:**

- 不重新训练模型，也不把部署 checkpoint 从 `avg_30.pt` 改为更新 epoch。
- 不修改 `wekws` 源码、上游 recipe 或通用导出链路的语义。
- 不把 Android 加载链路改成 `.ort` 或二次 softmax。
- 不在当前无 Android SDK 的环境里完成 APK 构建；只保证 host 对齐和脚本在 SDK 可用环境下可调用。

## Decisions

### 1. 用 `scripts/common/env.sh` 作为唯一路径来源

所有外层入口先 source `scripts/common/env.sh`，由它定义 `WORKSPACE_ROOT`、`WEKWS_ROOT`、`WORK`、config、data、dict、exp、tensorboard、exports 和下载目录，并设置 `PYTHONPATH`、`PYTHONIOENCODING`、`PYTORCH_CUDA_ALLOC_CONF`、`PYTHONDONTWRITEBYTECODE=1` 与 `PYTHONPYCACHEPREFIX="$WORK/pycache"`。

理由：避免在每个脚本里重复绝对路径，同时阻止 Python 在只读上游写 bytecode。备选方案是继续依赖 `path.sh` 和 cwd 相对路径，但这正是当前不可移植的根因，已放弃。

### 2. 大目录先迁移再重建 `data.list`

迁移前先生成清单（文件数、大小、sha256、epoch 序列、`avg_30.pt` 基线），并检测 `train.py` / `torchrun` 是否仍在写上游 `exp/`。无写入进程时在同一文件系统内原子 `mv`；有写入进程时先复制到 `work/` 校验，待写入结束后切换。所有目录迁移前后对比大小和文件数，验收前保留旧目录；`data.list` 不复用，必须由迁移后的音频路径重新生成。

理由：旧清单含指向上游 recipe 的绝对路径，简单文本替换容易漏掉变体且无法验证音频实际存在；`exp/` 还包含正在累积的 checkpoint 链和固定部署 `avg_30.pt`，直接移动有丢链或漂移风险。备选方案是 symlink 回上游目录，但这会继续污染 `wekws/`，因此放弃。

### 3. 配置副本放在外层，续训使用独立配置

`config/hi_xiaowen/ds_tcn_ctc.yaml` 承载当前有效的 `batch_size: 64`；`ds_tcn_ctc_resume.yaml` 独立承载降低后的学习率。上层训练/续训入口显式传入 config、dict、checkpoint、model_dir 和 tensorboard_dir，不再依赖修改后的上游 `run_ctc.sh`。

理由：上游必须恢复原样，`train.py` 又没有 `--lr`；把学习率放在外层独立 yaml 既能续训又不污染基础配置。备选方案是继续维护上游 diff，与只读约束冲突。

### 4. 实体脚本放入 `scripts/`，根目录和 android 目录保留薄 wrapper

实体实现按数据、训练、导出、评估、部署分目录放在 `scripts/`。根目录 `infer_cli.py` / `infer_webui.py` 保留为兼容 wrapper；`android/export/export_ctc_onnx.py` 与 `android/tests/run_compare.sh` 可保留调用入口，但默认值必须来自外层 env 或显式参数。

理由：现有命令和 Android 工程入口仍会被使用，直接硬切换会让旧调用静默读到上游路径。备选方案是删除所有旧入口，迁移成本高且不利于回归。

### 5. Android 资产从固定工作点成组导出

Android 导出以 `work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt` 为默认 checkpoint，`--keyword_thresholds` 必填且只接受 `docs/evaluation/avg30_frr_far_report.md` 记录的保守工作点 `嗨小问=0.303`、`你好问问=0.039`，一次生成 `kws.onnx`、`config.json`、`keywords.txt`、`id_to_token.json`。评测报告保留历史理论点 `0.297 / 0.000`；当前 3 位小数 `score.txt` 重算得到 `0.293 / 0.038`，两者都只用于评测记录，部署仍固定 `0.303 / 0.039`。导出后校验 token id、阈值、动态 batch 形状、posterior 输出和三方 score 对齐。

理由：assets 是部署事实来源；如果允许默认阈值或隐式新 checkpoint，早期高分路径和 FRR/FAR 工作点会漂移。备选方案是导出后 patch config，容易被遗漏且无法保证 ONNX 与 config 来自同一 checkpoint。

### 6. 用“备份 + 重新 clone”转换 submodule

先保存上游 diff 并恢复两处修改，确认下载物已迁出、`git -C wekws status --short` 为空；随后把旧目录改名为 `wekws.old`，重新 `git submodule add`，checkout 到 pin commit，验收完成后再删除备份。

理由：当前 `wekws/` 含本地修改、未跟踪大目录和嵌套 git repo，直接 absorb 风险高。备选方案是原地转换，但失败后更难回滚。

### 7. 分阶段验收门禁

按 Phase 1–5 推进，Phase 4 共 21 项验收：外层配置、默认 base checkpoint、dict token id、`data.list`、训练/续训、FRR/FAR、通用 ONNX/JIT、Android ONNX、CLI score、三方对齐、上游只读、`token_probs`、逐帧 top-k、`id_to_token`、阈值、`score_file` 格式、3 位小数 `score.txt`、`--step 0.001` 与 stats/det、`awk` 选点与报告、CMVN/cwd 文档约束，以及最终 git 边界。

理由：该迁移跨脚本、数据、模型和 git 结构，单点检查不足以保证等价。备选方案是先转 submodule 再补脚本，会在上游不干净时放大回滚成本。

## Risks / Trade-offs

- **29G 数据迁移耗时或中断** → 使用 `mv`，迁移前后记录 `du -sh` 与文件数；验收通过前保留旧目录。
- **训练产物迁移丢链或漂移** → 迁移前记录 epoch 序列、sha256 与 `avg_30.pt` 基线；检测写入进程，必要时先复制校验再切换，验收前保留旧目录可回退。
- **`data.list` 仍含旧绝对路径** → 强制重新生成并扫描 `wekws/examples/hi_xiaowen/s0/data/`；发现旧路径即失败。
- **dict token id 错位** → 从 `mobvoi_kws_transcription/tokens.txt` 生成，并校验 `你/嗨/好/小/问` 的 id 分别为 `1027/1302/1401/1462/2494`。
- **`score_ctc.py` CMVN 相对路径失败** → `eval_ctc.sh` 生成绝对路径评测 config 或固定可解析 cwd，禁止依赖旧 recipe 目录。
- **FRR/FAR 流程被简化** → 强制保留 `--step 0.001`、3 位小数 `score.txt`、`stats/*.txt`、`det.png`、`awk '$2 <= 0.2 {print; exit}'` 选点和报告模板。
- **`torchaudio` 版本不符** → `requirements.txt` 锁 `2.5.1`，env 检查 `torchaudio.info`；缺失时明确报错。
- **导出覆盖正确 assets** → 必填 `--keyword_thresholds`，固定 `avg_30.pt` 默认值，并在导出后检查 config、token 映射和 score 对齐。
- **debug 输出改变 score 语义** → `token_probs` 与 `frame_topk` 只走 debug 路径，`score_file` 格式不变，并用现有测试 wav 做回归。
- **上游转 submodule 失败** → 保留 `docs/plans/backups/wekws_local_changes.diff` 和 `wekws.old`，按 Phase 顺序回滚。
- **Android SDK 不可用** → 本地只跑 Native host 对齐；`build_apk.sh` 必须检测 SDK 并给出明确失败原因，不伪装成功。

## Migration Plan

1. **Phase 0**：记录 `wekws` commit 与 diff，备份到 `docs/plans/backups/wekws_local_changes.diff`，盘点现有产物，确认没有未知用户工作。
2. **Phase 1**：创建 `scripts/`、`config/hi_xiaowen/`、`work/` 子目录和 `scripts/common/env.sh`，复制外层 config，校验依赖版本。
3. **Phase 2**：生成训练产物清单并检测写入进程，迁移下载物和产物到 `work/`，重新生成 `data.list`，对比大小/文件数/sha256/epoch 链，验证前不删除原目录。
4. **Phase 3**：实现数据、训练/续训、评估、导出、推理和部署验证脚本，修正 `infer_core.py` 默认路径与 Android 默认值，更新文档。
5. **Phase 4**：执行 21 项等价验证，重点检查路径、token id、动态 batch ONNX、参数阈值、CLI/WebUI/Android 调试字段、三方 score 对齐、FRR/FAR 产物与 CMVN/cwd 约束。
6. **Phase 5**：恢复上游两处修改，确认 `wekws` 工作树为空，转换为 pin 住的 submodule，最终验收后删除 `wekws.old`。

回滚策略：任一阶段失败时停止后续阶段；数据和脚本问题从 `wekws.old` 与原路径恢复；assets 问题可用 `avg_30.pt` 和固定阈值重新导出；submodule 转换失败时移除新 submodule 并把 `wekws.old` 改回 `wekws`。
