# Design

## Context

本次变更只处理文档，动机见 `proposal.md`。当前事实来源已经存在且分散：外层 `scripts/` 是实际命令入口，`scripts/common/env.sh` 集中定义路径、Python、checkpoint 与阈值常量；`docs/manuals` 的训练手册保存环境与训练细节，`docs/evaluation` 的评测手册与报告保存 FRR/FAR 操作和结果；`android/app/src/main/assets/` 与导出脚本保存运行时契约。`wekws/` 是 pin 住的只读 submodule，上游 `run_ctc.sh` stage 只能作为历史说明。

## Goals / Non-Goals

**Goals:**

- 提供一份按当前外层脚本阶段组织的指导手册，使使用者无需搜索多个文档即可找到正确入口。
- 明确当前工作点、固定路径、产物位置和部署契约，避免误用历史 checkpoint、历史阈值或上游 recipe 路径。
- 在文档中完整保留 FRR/FAR 评测流程与 Android 三方对齐流程。
- 通过与现有手册互相链接，形成“总览指南 + 专题手册”的文档层级。

**Non-Goals:**

- 不修改任何脚本、配置、模型资产、评测产物或 submodule 内容。
- 不重跑训练、评估或导出，不用静态文档替代实际验证记录。
- 不把本机缺失 Android SDK 的环境限制写成 APK 已构建成功。
- 不新增或调整 OpenSpec 系统行为规格；本 change 使用 `skip_specs: true`。

## Decisions

1. **新建单一总览手册，而不是改写现有专题手册。**  
   新手册负责当前 `scripts/` 工作流导航、命令速查和固定约束；训练手册继续负责 Conda/显存/负样本等细节，FRR/FAR 手册继续负责评测原理与完整参数。这样避免大段重复内容在多个文件间漂移。替代方案是把所有内容塞进两份现有手册，但会让“当前入口”和“历史参考”混杂，维护成本更高。

2. **事实优先级固定为：脚本与 `env.sh` > 部署资产/评测报告 > 现有手册历史叙述。**  
   手册中的命令和默认值从当前脚本读取，阈值和 checkpoint 从 `env.sh`、导出脚本、assets 与评测报告交叉确认。若旧文档出现 `run_ctc.sh` stage、`15.pt` 联调模型或历史 FRR/FAR 选点，只在明确标注“历史记录/仅追溯”时保留。替代方案是照抄旧手册，风险是把已废弃参数 `--output_dir`、`--cmvn_file` 或旧 recipe 路径重新引入。

3. **手册以“仓库根目录命令”组织，每段给出前置条件、命令、产物和校验。**  
   阶段顺序为环境 → 数据 → 训练/续训 → 评估 → 导出 → 推理 → 部署验证。每个命令从 `source scripts/common/env.sh` 开始，明确 `$WORK`、`$DEPLOY_CHECKPOINT`、`$KEYWORD_THRESHOLDS` 等变量来自 `env.sh`，并说明所有产物写入 `work/`，不回写 `wekws/`。替代方案是按上游 stage 编号组织，但会误导使用者继续修改上游脚本。

4. **FRR/FAR 章节保留完整评测链和可复算公式。**  
   同时写明 `score.txt` 3 位小数、`--step 0.001`、`stats/*.txt` 与 `det.png`、`awk '$2 <= 0.2 {print; exit}'`、三条阈值集合：当前复算理论点 `0.293 / 0.038`、历史理论点 `0.297 / 0.000`、部署点 `0.303 / 0.039`。这样既不丢失评测流程，也不会把历史点误当部署配置。

5. **Android 部署章节以 assets 和导出脚本为唯一契约。**  
   明确直接加载 `kws.onnx`，不转换 `.ort`；动态输入 `[N,T,40]` / `[N,256,105]`、非 cache 输出 `[N,T,2599]`、后处理不再 softmax；score 沿用 `sqrt(∏ token_prob)`；`token_probs` 与 `frame_topk` 仅为调试。推理和 Native host 对齐命令引用外层入口，APK 构建只写前置条件与失败提示。

6. **交叉链接采用最小改动。**  
   只在训练手册和 FRR/FAR 评测手册顶部加入新指南链接，并保留原文细节。这样实施范围可控，且不会破坏历史记录的可追溯性。

## Risks / Trade-offs

- **[文档与脚本漂移]** → 每个命令实施时逐项对照当前脚本参数；在手册中注明“命令以 `scripts/` 实际参数为准”，并列出关键默认值来源 `scripts/common/env.sh`。
- **[历史命令误导]** → 对 `run_ctc.sh` stage、`15.pt`、`0.297 / 0.000`、`0.175212 / 3.31%` 等明确加“历史记录/仅追溯”标签，当前推荐命令只列外层入口。
- **[手册过长导致可用性下降]** → 采用“速查表 + 分阶段小节 + 专题链接”结构，详细环境与评测原理保留在专题手册，不在新指南中复制全文。
- **[阈值/score 语义被误解]** → 用独立小节说明 per-keyword 阈值、无全局 `ctc.trigger_threshold`、score 公式和调试字段边界，并注明阈值不改变 score 或 `score_file` 格式。
- **[APK 构建状态被误读]** → 写明本机无 Android SDK 时 `build_apk.sh` 会失败，只完成 Native host 与三方对齐；不把文档说明当作构建证据。
- **[文档-only change 与 OpenSpec 预期不符]** → 通过 `.openspec.yaml` 的 `skip_specs: true` 明确无系统行为变化，避免为验证形式编造需求。

## Migration Plan

1. 新增 `docs/manuals/script_workflow_guide.md`，按上述阶段和固定约束写入当前工作流。
2. 从 `scripts/common/env.sh`、各 `scripts/**` 入口、`android/tests/run_compare.sh`、`docs/evaluation/avg30_frr_far_report.md` 复核命令、路径、阈值和历史/当前数据标注。
3. 在现有训练手册与 FRR/FAR 评测手册顶部加入新指南链接，不改动其正文事实。
4. 审查链接路径、代码块参数和文档内自洽性，确保没有指向上游 recipe 的推荐命令。
5. 提交时只包含文档变更；如需回退，直接 revert 本次文档提交即可，不影响模型或实验产物。
