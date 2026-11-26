# Tasks

## 1. 事实盘点与文档骨架

- [x] 1.1 盘点 `scripts/common/env.sh` 与 `scripts/01_data`、`scripts/02_train`、`scripts/03_export`、`scripts/04_eval`、`scripts/05_deploy` 的实际入口、参数、默认值和产物路径；在手册中形成命令速查表，并逐项对照脚本源码确认无臆测参数。
- [x] 1.2 建立 `docs/manuals/script_workflow_guide.md` 骨架，包含“当前工作流总览、环境、数据、训练/续训、评估、导出、推理、部署验证、故障排查”章节；以文件存在且章节覆盖了 proposal 中列出的全部阶段为完成标准。
- [x] 1.3 在手册开头写明事实优先级“脚本与 `env.sh` > 部署资产/评测报告 > 历史手册”，并明确上游 `run_ctc.sh` stage、`15.pt` 与历史阈值只作追溯；以相关历史字样均带有“历史/仅追溯”标注为完成标准。

## 2. 编写阶段命令与固定约束

- [x] 2.1 编写环境与数据章节：记录 `source scripts/common/env.sh`、`bash scripts/common/check_deps.sh`、两个下载脚本、`prepare_mobvoi.sh`、`regenerate_data_lists.sh`、`prepare_dict.sh`；以命令与脚本参数一致、产物均写到 `work/` 为完成标准。
- [x] 2.2 编写训练与续训章节：记录 `train_ctc.sh` 的 base checkpoint `23.pt`、外层 config、`num_keywords=2599`、`model_dir`/`tensorboard_dir`，以及 `resume_ctc.sh --checkpoint`、resume config、独立输出目录和拒绝 `init.pt`；以手册不出现 `train.py --lr` 或未显式 `--dict` 的推荐命令为完成标准。
- [x] 2.3 编写评估章节：记录 `eval_ctc.sh`、average fallback、绝对 CMVN config、ASCII unicode escape 关键词、3 位小数 `score.txt`、`--step 0.001`、`stats/*.txt`、`det.png`、`awk '$2 <= 0.2 {print; exit}'` 与报告模板；以完整覆盖 FRR/FAR 手册中的可复现流程为完成标准。
- [x] 2.4 在评估章节明确区分当前复算理论点 `0.293 / 0.038`、历史理论点 `0.297 / 0.000` 和部署点 `0.303 / 0.039`；以三组数值都有明确语义且部署阈值不被历史值替代为完成标准。
- [x] 2.5 编写导出章节：记录通用 ONNX/JIT 的 `--onnx_model` / `--jit_model`，以及 Android 固定导出命令 `$PYTHON_BIN scripts/03_export/export_android_onnx.py --keyword_thresholds '嗨小问=0.303,你好问问=0.039'`；以命令与脚本默认值一致、明确缺失阈值会失败为完成标准。
- [x] 2.6 编写 Android 部署契约：直接加载 `kws.onnx`、不转 `.ort`、动态 batch `[N,T,40]`/`[N,256,105]` -> `[N,T,2599]`、输出 posterior 且不二次 softmax、score 为 `sqrt(∏ token_prob)`、per-keyword 独立阈值、无全局 `ctc.trigger_threshold`、`token_probs`/`frame_topk` 仅调试；以每项约束在手册中有明确说明为完成标准。
- [x] 2.7 编写推理与部署验证章节：记录 CLI/WebUI 复用 `infer_core.py`、CLI 使用 `--detail_file`/`--dump_frame_topk`/`--score_file`、`avg_30.pt` 显式选择、`build_android_host.sh`、`compare_scores.sh` 与 APK 构建前置条件；以本机无 SDK 时只写前置条件、不声称 APK 构建成功为完成标准。
- [x] 2.8 编写故障排查表：至少覆盖 torchaudio 版本、`torchaudio.info`、sox、旧 `data.list` 路径、相对 CMVN、unicode escape 关键词、checkpoint 不足 30、GPU OOM、续训 epoch 重记、Android SDK 缺失；以每条都能对应现有脚本或专题手册事实为完成标准。

## 3. 专题文档交叉链接

- [x] 3.1 在 `docs/manuals/wekws + MobvoiHotwords 完整训练手册（Conda 环境）.md` 顶部加入新指南链接；以链接可解析、原文环境和训练细节未被删改为完成标准。
- [x] 3.2 在 `docs/evaluation/avg30_frr_far_evaluation.md` 顶部加入新指南链接；以链接可解析、FRR/FAR 原流程和历史记录未被覆盖为完成标准。

## 4. 一致性与范围校验

- [x] 4.1 对新指南执行静态自查：确认推荐命令只使用外层 `scripts/` 入口，且不含 `--output_dir`、`--cmvn_file`、推荐性 `run_ctc.sh`、全局 `ctc.trigger_threshold` 或 `.ort` 转换；以 grep/人工复核无命中为完成标准。
- [x] 4.2 检查所有仓库内相对链接、代码块参数与数值标注，确认三类 FRR/FAR 阈值和部署 score 对齐数据一致；以 `git diff --check` 通过且链接目标存在为完成标准。
- [x] 4.3 确认变更范围只包含 `docs/manuals/script_workflow_guide.md` 与两份专题手册链接；以 `git status --short` 中没有 `scripts/`、`wekws/`、`work/`、模型资产或配置变更作为完成标准。
- [x] 4.4 提交本次文档变更，并在提交信息中说明仅新增/链接当前脚本工作流指南；以提交包含且仅包含上述文档文件为完成标准（commit `c058938`）。

## Workflow follow-up

- 在项目评审要求满足后归档本 change。
- 归档后核对 main specs 无变化，确认该文档变更未引入系统行为 delta。
