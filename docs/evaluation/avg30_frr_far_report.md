# avg_30.pt FRR/FAR 评测结果

> 生成时间：2026-11-26
> checkpoint：`work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`（sha256 `fa02f745cb48fdb961893981e5bd7e8836738e47569ea7997a723ed09e12149c`）
> score 文件：`work/eval/avg_30_pt/score.txt`（3 位小数）
> DET 参数：`--step 0.001`、`--window_shift 50`
> stats：`work/eval/avg_30_pt/stats/stats.嗨_小_问.txt`、`stats.你_好_问_问.txt`
> DET 图：`work/eval/avg_30_pt/stats/det.png`

## 评测规模与模型平均

- 固定部署 checkpoint `avg_30.pt` 生成时打印的入选 epoch（历史记录，训练目录现已跑完 80 epoch）：`15, 16, 20, 22-57`。本次评测未重新生成或替换该 checkpoint。
- test 集：`work/data/hi_xiaowen/test/data.list`，共 `73459` 条 wav，总时长约 `74.131` 小时。
- 嗨小问正样本 `10641` 条（约 `5.512` 小时），负样本 `62818` 条（约 `68.618` 小时）。
- 你好问问正样本 `10641` 条，其余为负样本。

## 当前复算的理论最低工作点（`awk '$2 <= 0.2 {print; exit}'`）

以当前 `work/eval/avg_30_pt/score.txt` 重跑 `compute_det_ctc.py` 后的首个满足 `FAR ≤ 0.2/h` 的行：

| 关键词 | threshold | FAR（次/小时） | FRR |
|---|---:|---:|---:|
| 嗨小问 | 0.293 | 0.189454 | 3.947% |
| 你好问问 | 0.038 | 0.189813 | 3.2986% |

历史记录中曾使用 `嗨小问=0.297`、`你好问问=0.000` 作为理论最低点（FRR 约 `3.96%` / `3.20%`）。该数字来自早期评测，与当前 3 位小数 `score.txt` 的复算结果不同；保留作追溯，**不作为当前选点依据**。部署阈值没有因此改变。

## 部署工作点（留余量）

| 模型 | 关键词 | FAR 目标（次/小时） | 阈值 | FAR（次/小时） | FRR |
|---|---|---:|---:|---:|---:|
| avg_30.pt | 嗨小问 | 0.2 | 0.303 | 0.174880 | 4.0598% |
| avg_30.pt | 你好问问 | 0.2 | 0.039 | 0.189813 | 3.2986% |

历史报告中记录的部署 FAR/FRR 为 `0.174880 / 4.03%`（嗨小问）与 `0.175212 / 3.31%`（你好问问），同样来自早期评测；当前 stats 对应行分别是 `0.174880 / 4.0598%` 与 `0.189813 / 3.2986%`。以当前 stats 为准，部署阈值仍为 `0.303 / 0.039`：嗨小问保留约 `0.025/h` FAR 余量，你好问问保留约 `0.010/h` 余量，均满足 `FAR ≤ 0.2/h`。per-keyword 阈值彼此独立，同一 channel 可以同时上报两个关键词事件。

## 复现命令

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh
bash scripts/04_eval/eval_ctc.sh --checkpoint "$DEPLOY_CHECKPOINT" --result_dir "$RESULT_DIR"

awk '$2 <= 0.2 {print; exit}' "$RESULT_DIR/stats/stats.嗨_小_问.txt"
awk '$2 <= 0.2 {print; exit}' "$RESULT_DIR/stats/stats.你_好_问_问.txt"
```

评测脚本会派生带绝对 CMVN 路径的 `$RESULT_DIR/config.yaml`，不依赖旧 recipe cwd。`score_ctc.py` 无 `--cmvn_file` 参数，手工调用时必须使用该派生 config 或固定到可解析的相对路径 cwd。
