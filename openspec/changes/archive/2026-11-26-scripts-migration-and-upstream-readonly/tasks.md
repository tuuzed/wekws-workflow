# Tasks

## 1. Phase 0：基线与备份

- [x] 1.1 记录 `wekws` 当前 commit 与 diff，写入 `docs/plans/backups/wekws_local_changes.diff`；验证备份文件存在且包含 `ds_tcn_ctc.yaml` 与 `run_ctc.sh` 两处改动。
- [x] 1.2 盘点 `wekws/examples/hi_xiaowen/s0/` 下 `data/`、`dict/`、`exp/`、`tensorboard/`、`mobvoi_kws_transcription/` 的大小、文件数和路径清单，并记录 epoch 序列与 `avg_30.pt` 的 sha256；验证基线清单已保存，迁移前未删除原目录。
- [x] 1.3 记录当前部署资产基线：`kws.onnx`、`config.json`、`id_to_token.json` 的 sha256 与两个 per-keyword 阈值；验证 `config.json` 为 `0.303 / 0.039` 且不含全局 `ctc.trigger_threshold`。

## 2. Phase 1：外层结构、环境与配置

- [x] 2.1 创建 `scripts/common/env.sh`，集中定义 `WORKSPACE_ROOT`、`WEKWS_ROOT`、`WORK`、config/data/dict/exp/tensorboard/exports/下载目录及 Python 环境；验证 source 后变量正确，且 `PYTHONDONTWRITEBYTECODE=1`、`PYTHONPYCACHEPREFIX=$WORK/pycache` 生效。
- [x] 2.2 创建 `work/` 及 `pycache/`、`downloads/`、`data/`、`dict/`、`tensorboard/`、`exports/`、`exp/` 子目录，并核对 `.gitignore`；验证 `git check-ignore work/...` 命中，同时 `android/app/src/main/assets/kws.onnx` 仍被 git 跟踪。
- [x] 2.3 创建 `config/hi_xiaowen/ds_tcn_ctc.yaml` 与 `ds_tcn_ctc_resume.yaml`；验证基础配置含 `batch_size: 64`，resume 配置含 `lr: 0.0005`，且上游 yaml 未被改动。
- [x] 2.4 在 env 或前置检查中加入依赖版本校验；验证 `/root/miniconda/envs/wekws/bin/python` 下 `torch==2.5.1`、`torchaudio==2.5.1`、`torchaudio.info` 和 `sox` 均可用，不满足时输出明确错误。

## 3. Phase 2：下载资源与产物迁移

- [x] 3.1 把完整 `mobvoi_kws_transcription/` 移到 `work/downloads/mobvoi_kws_transcription/`；验证其 `.git` 存在、`23.pt` 与 `tokens.txt` 可用，且上游 recipe 下旧目录不存在。
- [x] 3.2 平稳迁移 `data/`、`dict/`、`exp/`、`tensorboard/` 到 `work/` 对应目录；迁移前检测 `train.py`/`torchrun` 写入进程，无写入时同文件系统原子 `mv`，有写入时先复制校验再切换；验证前后 `du -sh`、文件数、sha256、epoch 序列与 `avg_30.pt` 基线一致，验收前保留旧目录可回退。
- [x] 3.3 从迁移后音频重新生成 `train/dev/test/data.list`；验证所有 wav 路径位于 `work/` 且实际存在，`grep -R 'wekws/examples/hi_xiaowen/s0/data' work/data/hi_xiaowen` 无命中。
- [x] 3.4 从 `work/downloads/mobvoi_kws_transcription/tokens.txt` 生成 `work/dict/hi_xiaowen/dict.txt` 与 `words.txt`；验证 `你/嗨/好/小/问` 对应 `1027/1302/1401/1462/2494`，且 `&` 已替换为 `<filler>`。

## 4. Phase 3：外层工作流脚本

- [x] 4.1 实现 `scripts/01_data/download_mobvoi.sh` 与 `download_mobvoi_transcription.sh`；验证目标目录不存在时下载到 `work/downloads/`，存在时跳过且不修改已有资源。
- [x] 4.2 实现 `prepare_mobvoi.sh`，调用上游数据准备工具但全部输出到 `work/`；验证 wav.scp/text/wav.dur/data.list/global_cmvn 均生成在 `work/data/hi_xiaowen/`，且 `wekws/` 内没有新文件。
- [x] 4.3 实现 `train_ctc.sh`，显式传 config、dict、train/dev data.list、model_dir、tensorboard_dir、cmvn 和 checkpoint；验证默认 checkpoint 为 `work/downloads/mobvoi_kws_transcription/23.pt`，显式 `--checkpoint` 可覆盖。
- [x] 4.4 实现 `resume_ctc.sh`，使用 resume config 和独立输出目录并拒绝 `init.pt`；验证 resume 配置学习率低于基础配置，传入 `init.pt` 时命令失败且不启动训练。
- [x] 4.5 实现 `eval_ctc.sh`，包含平均模型 fallback、score 与 DET、FRR/FAR 选点和报告；验证 checkpoint 数 <30 时生成 `avg_N.pt` 或使用 `final.pt`，`score.txt` 为 3 位小数、DET 使用 `--step 0.001`、stats/det 落在 `work/eval/`，显式传 `--dict` 与 `\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee`，并按 `awk '$2 <= 0.2 {print; exit}'` 记录当前复算点 `0.293 / 0.038` 与历史理论点 `0.297 / 0.000`，部署仍为 `0.303 / 0.039`。
- [x] 4.6 实现 `export_onnx.sh` 与 `export_jit.sh`，保留通用导出链路；验证分别使用 `--onnx_model` / `--jit_model`，产物位于 `work/exports/hi_xiaowen/`。
- [x] 4.7 把 Android 导出实体迁移到 `scripts/03_export/export_android_onnx.py`，保留旧路径薄 wrapper；验证默认 checkpoint 为 `work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`，缺少 `--keyword_thresholds` 时失败。
- [x] 4.8 修正 `android/tests/run_compare.sh` 默认 checkpoint；验证默认指向 `work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`，且 `CHECKPOINT` 环境变量仍可覆盖。
- [x] 4.9 修改 `infer_core.py` 路径常量并把 CLI/WebUI 实体放入 `scripts/04_eval/`，根目录保留 wrapper；验证默认 config/dict/checkpoint/score_file 均指向 `work/`，不再指向上游 `exp/`。
- [x] 4.10 实现 `build_android_host.sh`、`compare_scores.sh`、`build_apk.sh`；验证 host 对齐可通过外层脚本执行，未检测到 Android SDK 时 `build_apk.sh` 明确失败且不伪装成功。
- [x] 4.11 更新 `docs/README.md`、`docs/manuals/` 和 `docs/plans/` 中的推荐命令为新 `scripts/` 路径；验证文档命令均可按写出的形式解析，且不再推荐上游 `run_ctc.sh`。

## 5. Phase 3：部署资产、FRR/FAR 流程与文档清理

- [x] 5.1 用 `avg_30.pt` 与 `嗨小问=0.303,你好问问=0.039` 重新导出 Android 四项资产；验证 `config.json` 两个阈值正确、不含 `ctc.trigger_threshold`，token id 分别为 `[1302,1462,2494]` 与 `[1027,1401,2494,2494]`。
- [x] 5.2 验证导出的 ONNX 形状与输出语义；验证输入为 `[N,T,40]` / `[N,256,105]`，输出为 `[N,T,2599]` / `[N,256,105]`，且 PyTorch 与 ORT posterior 最大差在既定容差内。
- [x] 5.3 验证 `token_probs`、`frame_topk` 与 `--score_file` 不变；验证 CLI/WebUI/Android 事件均包含 token 概率，`sqrt(∏ token_probs) == score`，默认 top-k 关闭且原 score 文件格式保持不变。
- [x] 5.4 清理 `android/README.md`、`AGENTS.md`、`docs/evaluation/avg30_frr_far_evaluation.md` 中旧 `15.pt` / “联调 ONNX” / “脚本尚不存在” 的表述，并补充 CMVN 绝对路径或固定 cwd 约束与 `avg30_frr_far_report.md` 报告模板；验证 `grep -R '联调 ONNX\|15.pt\|只是计划项' android/README.md AGENTS.md docs` 仅保留明确标注为历史记录或已删除语境的结果。

## 6. Phase 4：等价验证门禁

- [x] 6.1 校验外层训练配置；验证 `config/hi_xiaowen/ds_tcn_ctc.yaml` 含 `batch_size: 64`、`num_keywords: 2599`，且上游 yaml 未改动。
- [x] 6.2 校验默认 base checkpoint；验证未传 `--checkpoint` 时训练入口使用 `work/downloads/mobvoi_kws_transcription/23.pt`。
- [x] 6.3 校验 dict token id；验证 `work/dict/hi_xiaowen/dict.txt` 中 `你/嗨/好/小/问` 对应 `1027/1302/1401/1462/2494`，且 `&` 已替换为 `<filler>`。
- [x] 6.4 校验迁移后数据清单与训练产物链；验证 `work/data/hi_xiaowen/{train,dev,test}/data.list` 的 wav 全部位于 `work/`，且 `work/exp/hi_xiaowen/ds_tcn_ctc/` 的 epoch 序列、sha256、`avg_30.pt` 与基线一致。
- [x] 6.5 验证 `train_ctc.sh` 可启动训练；验证产物写 `$EXAMPLE_EXP`、tensorboard 写 `$EXAMPLE_TB`，并确认该目录已完成完整 80 epoch 训练（`0.pt`–`79.pt`、`avg_30.pt` 齐全）；迁移过程有写入进程时采用复制校验后切换且旧目录保留可回退。
- [x] 6.6 验证 `resume_ctc.sh` 使用 resume config；验证 `lr: 0.0005`、输出独立目录，传入 `init.pt` 时命令失败且不启动训练。
- [x] 6.7 验证 `eval_ctc.sh` 产出 FRR/FAR；验证 `avg_N.pt`/`final.pt` fallback，并在 `FAR ≤ 0.2/h` 下复现 `嗨小问=0.303`、`你好问问=0.039`。
- [x] 6.8 验证 `export_onnx.sh` 使用 `--onnx_model`，且在 `work/exports/hi_xiaowen/` 生成通用 ONNX。
- [x] 6.9 验证 `export_jit.sh` 使用 `--jit_model`，且在 `work/exports/hi_xiaowen/` 生成 JIT 模型。
- [x] 6.10 验证 `export_android_onnx.py` 从 `avg_30.pt` 导出动态 batch `kws.onnx`；验证输入 `[N,T,40]`/`[N,256,105]`、输出 `[N,T,2599]`/`[N,256,105]`。
- [x] 6.11 验证 `infer_cli.py` 对 `ch0.wav`、`ch_nihao.wav` 的 score 与 `avg_30.pt` PyTorch 参考一致（旧 ONNX 资产偏差约 `0.008`，重导出后修复），且默认路径不再指向上游 `exp/`。
- [x] 6.12 运行 CLI/ONNX/Native 三方 score 对齐；验证 `ch0.wav`、`ch_nihao.wav`、`stereo.wav` 的 score 一致，检测事件分别为 `嗨小问`、`你好问问`、仅 channel 0 命中。
- [x] 6.13 执行上游只读校验；验证 `wekws/` 内不存在 `data/`、`dict/`、`exp/`、`tensorboard/`、`mobvoi_kws_transcription/`、`__pycache__/`，且 `git -C wekws status --short` 为空。
- [x] 6.14 验证 CLI/WebUI/Android 事件 JSON 均含 `token_probs`，且 `sqrt(prod(token_probs)) == score`。
- [x] 6.15 验证 `--dump_frame_topk` 与 `frameDebug()` 的 token id 与 ORT 参考一致，且默认关闭时输出为空。
- [x] 6.16 验证 `id_to_token.json` 正确映射关键词 token `1302/1462/2494` 与 `1027/1401/2494/2494`。
- [x] 6.17 验证 `android/app/src/main/assets/config.json` 的 per-keyword 阈值为 `0.303 / 0.039`，且不含 `ctc.trigger_threshold`。
- [x] 6.18 验证 `--score_file` 原格式保持不变，且 `score.txt` 可被 `compute_det_ctc.py` 直接读取。
- [x] 6.19 验证 `score.txt` 保留 3 位小数、DET 使用 `--step 0.001`，且 `stats.嗨_小_问.txt`、`stats.你_好_问_问.txt` 与 `det.png` 位于 `$RESULT_DIR/stats`。
- [x] 6.20 验证 `awk '$2 <= 0.2 {print; exit}'` 在当前 `score.txt` 上复算得到 `0.293 / 0.038`，报告同时保留历史理论点 `0.297 / 0.000`，部署实际使用留余量点 `0.303 / 0.039`。
- [x] 6.21 验证 `docs/evaluation/avg30_frr_far_evaluation.md` 已改为 `scripts/` / `$WORK` 入口，仍明确说明 `score_ctc.py` 的 CMVN 相对路径约束，且复核外层 git 只含项目文件与 submodule 指针。

## 7. Phase 5：恢复上游并转 submodule

- [x] 7.1 恢复 `wekws/examples/hi_xiaowen/s0/conf/ds_tcn_ctc.yaml` 与 `run_ctc.sh`；验证两文件与上游 pin commit 一致，本地改动只保留在 `docs/plans/backups/wekws_local_changes.diff`。
- [x] 7.2 按“备份 + 重新 clone”方案把 `wekws` 转为 submodule 并 pin `6a45aeb994dd81c0969ff877a5a7c46d60ed0c86`；验证 `.gitmodules` URL 为 `https://github.com/wenet-e2e/wekws.git`，且 `git submodule status` 显示该 commit。
- [x] 7.3 最终验收通过后删除 `wekws.old`；验证清理后 `git status --short` 与 `git -C wekws status --short` 均满足只读与提交边界。
- [x] 7.4 更新 `AGENTS.md` 与计划文档状态为迁移完成；验证文档不再把 `wekws/` 描述为普通目录，也不再声明 `scripts/`、`config/`、`env.sh` 不存在。

## Workflow follow-up

- 在评审通过后提交外层仓库变更，不提交 `wekws` 源码。
- Archive 此 change 并确认生成的 main spec 与 delta 一致。
