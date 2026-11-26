# Proposal

## Why

当前 `docs/manuals` 的训练手册和 `docs/evaluation` 的 FRR/FAR 手册分别记录了完整训练细节与评测操作，但缺少一份按外层 `scripts/` 实际入口组织的当前工作流总览；使用者容易回到上游 `run_ctc.sh` stage 或旧路径。迁移已完成、`wekws/` 已 pin 为只读 submodule，现在需要把分散在手册、脚本、资产和评测报告中的当前事实收口成可执行的指导手册。

## What Changes

- 新增 `docs/manuals/script_workflow_guide.md`，作为当前外层脚本工作流的单一入口手册，明确所有推荐命令从仓库根目录调用 `scripts/`。
- 手册按阶段覆盖：环境检查、下载与数据准备、训练/续训、FRR/FAR 评估、通用 ONNX/JIT 导出、Android 资产导出、CLI/WebUI 推理、Native host 构建与三方 score 对齐、APK 构建前置条件。
- 汇总 `scripts/common/env.sh` 中的权威路径与默认值，并明确固定约束：部署 checkpoint 为 `work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`，阈值为 `嗨小问=0.303`、`你好问问=0.039`，无全局 `ctc.trigger_threshold`。
- 完整保留 FRR/FAR 流程要点：`score.txt` 3 位小数、`--step 0.001`、`stats/*.txt`、`det.png`、`awk '$2 <= 0.2 {print; exit}'`、报告模板；区分当前复算点 `0.293 / 0.038`、历史理论点 `0.297 / 0.000` 与部署点 `0.303 / 0.039`。
- 写明 Android 直接加载 `kws.onnx`、动态 batch `[N,T,40]`/`[N,256,105]` -> `[N,T,2599]`、输出 posterior 且不二次 softmax、score 为 `sqrt(∏ token_prob)`、`token_probs`/`frame_topk` 仅用于调试。
- 增加常见失败与恢复指引，包括相对 CMVN 路径、unicode escape 关键词、checkpoint 数量不足、GPU OOM、续训 epoch 重记、Android SDK 缺失。
- 在现有训练手册与 FRR/FAR 评测手册中加入指向新指南的入口，并保留旧 `run_ctc.sh` stage 只作历史参考的说明。
- 不修改脚本、配置、模型资产、评测产物或 `wekws/` submodule；本 change 只新增/链接文档。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

无。本 change 仅新增和链接文档，不改变系统可观察行为；`.openspec.yaml` 设置 `skip_specs: true`，不新增或调整 delta spec。

## Impact

- 文档：新增 `docs/manuals/script_workflow_guide.md`；更新 `docs/manuals/wekws + MobvoiHotwords 完整训练手册（Conda 环境）.md` 与 `docs/evaluation/avg30_frr_far_evaluation.md` 的引导链接。
- 代码/API：无变化；CLI/WebUI、导出脚本、评测脚本和 Native 校验入口保持原样。
- 依赖：无新增依赖；仍使用 conda 环境 `/root/miniconda/envs/wekws/bin/python`、`torch`/`torchaudio` 2.5.1 与 conda-forge `sox`。
- 已知限制：本机无 Android SDK，手册只记录 APK 构建前置条件，不把 APK 构建标记为已验证。
