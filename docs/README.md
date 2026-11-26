# 文档索引

`docs/` 按文档类型分类：

| 分类 | 目录 | 用途 |
|---|---|---|
| 架构与设计 | `architecture/` | 系统架构、score 计算、自定义唤醒词支持等设计说明 |
| 评测 | `evaluation/` | FRR / FAR 评测、阈值选择与结果记录 |
| 操作手册 | `manuals/` | 环境与训练流程等可照着执行的手册 |
| 实施计划 | `plans/` | 迁移、整理和阶段性执行计划 |

## 常用入口

所有命令从仓库根目录执行，路径与环境由 `scripts/common/env.sh` 统一提供：

```bash
source scripts/common/env.sh
bash scripts/common/check_deps.sh
bash scripts/01_data/prepare_mobvoi.sh
bash scripts/02_train/train_ctc.sh --checkpoint "$BASE_CHECKPOINT"
bash scripts/02_train/resume_ctc.sh --checkpoint "$WORK/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt"
bash scripts/04_eval/eval_ctc.sh
$PYTHON_BIN scripts/03_export/export_android_onnx.py --keyword_thresholds '嗨小问=0.303,你好问问=0.039'
bash scripts/05_deploy/build_android_host.sh
bash scripts/05_deploy/compare_scores.sh android/tests/data/ch0.wav
```

## 架构与设计

- [Android 端多路 KWS-CTC 部署架构设计与实现](architecture/android_deployment_architecture.md)
- [CTC 推理 Score 的计算方式](architecture/ctc_inference_score.md)
- [自定义唤醒词支持设计](architecture/custom_keywords_support.md)

## 评测

- [avg_30.pt FRR/FAR 评测操作手册](evaluation/avg30_frr_far_evaluation.md)
- [avg_30.pt FRR/FAR 评测结果](evaluation/avg30_frr_far_report.md)

## 操作手册

- [wekws + MobvoiHotwords 完整训练手册（Conda 环境）](<manuals/wekws + MobvoiHotwords 完整训练手册（Conda 环境）.md>)

## 实施计划

- [脚本归置与 wekws 上游只读迁移计划](plans/scripts_migration_and_upstream_readonly_plan.md)
- [独立关键词阈值方案实施计划](plans/independent_keyword_threshold_plan.md)
