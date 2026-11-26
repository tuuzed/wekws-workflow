# 文档索引

`docs/` 按文档类型分类：

| 分类 | 目录 | 用途 |
|---|---|---|
| 架构与设计 | `architecture/` | 系统架构、score 计算、自定义唤醒词支持等设计说明 |
| 评测 | `evaluation/` | FRR / FAR 评测、阈值选择与结果记录 |
| 操作手册 | `manuals/` | 环境与训练流程等可照着执行的手册 |
| 实施计划 | `plans/` | 迁移、整理和阶段性执行计划 |

## 架构与设计

- [Android 端多路 KWS-CTC 部署架构设计与实现](architecture/android_deployment_architecture.md)
- [CTC 推理 Score 的计算方式](architecture/ctc_inference_score.md)
- [自定义唤醒词支持设计](architecture/custom_keywords_support.md)

## 评测

- [avg_30.pt FRR/FAR 评测操作手册](evaluation/avg30_frr_far_evaluation.md)

## 操作手册

- [wekws + MobvoiHotwords 完整训练手册（Conda 环境）](<manuals/wekws + MobvoiHotwords 完整训练手册（Conda 环境）.md>)

## 实施计划

- [脚本归置与 wekws 上游只读迁移计划](plans/scripts_migration_and_upstream_readonly_plan.md)
- [独立关键词阈值方案实施计划](plans/independent_keyword_threshold_plan.md)
