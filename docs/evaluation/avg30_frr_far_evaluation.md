# avg_30.pt FRR/FAR 评测操作手册

> 当前外层 `scripts/` 工作流的命令速查与固定约束见[当前脚本工作流指导手册](../manuals/script_workflow_guide.md)。本文继续保留 FRR/FAR 评测参数、选点方法与历史记录。

> 日期：2026-11-26  
> 适用 checkpoint：`work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`  
> 关键词：`嗨小问`、`你好问问`  
> 工作点约束：FAR ≤ 0.2 次/小时  
> 说明：本文使用外层 `scripts/` / `$WORK` 路径；上游 recipe 保持只读。

---

## 1. 目的

用 `avg_30.pt` 在 test 集上计算每个唤醒词的：

- FRR（False Rejection Rate，漏唤醒率）
- FAR（False Alarm Rate，误唤醒率，按每小时误触发次数统计）
- DET 曲线（FRR vs FAR）

最终从 DET 结果中为每个关键词独立选取 Android 端 `keywords[].trigger_threshold` 工作点。

---

## 2. 前置条件

1. 已完成至少 30 个 epoch，实验目录包含 `0.pt` ~ `N.pt`（当前 epoch 57 已满足）。
2. 使用 conda 环境：

   ```bash
   /root/miniconda/envs/wekws/bin/python
   ```

3. 依赖 `matplotlib`、`pypinyin` 可用于绘制 DET 曲线。
4. test 集路径：

   ```text
   work/data/hi_xiaowen/test/data.list
   ```

5. 评测使用单卡 GPU。当前训练仍占用 GPU，若同时运行，需降低 batch size 或等待训练结束。

---

## 3. 环境设置

```bash
cd /workspace/wekws-my
source scripts/common/env.sh   # 统一设置 $WORK / $PYTHON_BIN / PYTHONPATH / pycache
```

推荐直接执行外层入口，它会自动生成带**绝对 CMVN 路径**的评测 config，因此不再依赖旧的 recipe cwd：

```bash
bash scripts/04_eval/eval_ctc.sh --checkpoint "$DEPLOY_CHECKPOINT" \
  --result_dir "$RESULT_DIR"
```

若手工调用 `score_ctc.py`，不得依赖旧 recipe cwd：要么把 `model.cmvn.cmvn_file` 改为绝对路径（例如 `$EXAMPLE_DATA/train/global_cmvn`，可用 `scripts/common/make_abs_cmvn_config.py` 生成），要么固定 cwd 到能解析 `data/train/global_cmvn` 的目录。训练产物中的 `config.yaml` 仍保留相对路径，不要直接拿它当评测 config。

---

## 4. 生成 avg_30.pt

```bash
source scripts/common/env.sh
$PYTHON_BIN -m wekws.bin.average_model \
  --dst_model "$DEPLOY_CHECKPOINT" \
  --src_path "$EXAMPLE_EXP" \
  --num 30 \
  --val_best
```

`--val_best` 会读取各 epoch 的 `*.yaml`，按 `cv_loss` 从低到高选前 30 个 checkpoint 做参数平均。

注意：

1. `config.yaml` 与 `init.yaml` 会被 `average_model.py` 的 glob 规则排除，不参与选择；
2. 若训练仍在写新 yaml，结果可能受竞态影响；要完全可复现，等训练停止后再执行；
3. 程序会打印 `selected epochs`，应确认输出正常。

当前 epoch 57 时，入选 epoch 大致为：

```text
15, 16, 20, 22-57
```

---

## 5. 生成 score 文件

手工调用时必须显式传 `--dict`，且 config 必须是带绝对 CMVN 路径的派生文件（`$RESULT_DIR/config.yaml`）：

```bash
source scripts/common/env.sh
mkdir -p "$RESULT_DIR/stats"
$PYTHON_BIN "$WEKWS_ROOT/wekws/bin/score_ctc.py" \
  --config "$RESULT_DIR/config.yaml" \
  --dict "$EXAMPLE_DICT" \
  --test_data "$EXAMPLE_DATA/test/data.list" \
  --gpu 0 \
  --checkpoint "$DEPLOY_CHECKPOINT" \
  --batch_size 32 \
  --num_workers 8 \
  --score_file "$RESULT_DIR/score.txt" \
  --keywords '\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee'
```

也可以直接 `bash scripts/04_eval/eval_ctc.sh`，它会一次性完成 config 派生、score、DET 和报告。

参数说明：

| 参数 | 说明 |
|---|---|
| `--dict` | 必须显式传，否则默认 `./dict` |
| `--keywords` | 必须传 ASCII unicode escape，不能传中文字面量 |
| `--batch_size` | 独占 GPU 可用 64；与训练同时跑建议 16/32 |
| `--score_file` | 输出每条 wav 的命中结果和 score |

`--keywords` 必须写成：

```bash
'\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee'
```

因为 `score_ctc.py` 内部会执行：

```python
args.keywords.encode('utf-8').decode('unicode_escape')
```

直接传中文会产生 mojibake。

`score.txt` 格式：

```text
<key> detected <关键词> <score>
<key> rejected
```

score 写入时只保留 3 位小数，因此后续阈值步长使用 `0.001`，更细没有意义。

---

## 6. 计算 FRR/FAR 与 DET 曲线

```bash
source scripts/common/env.sh
mkdir -p "$RESULT_DIR/stats"
$PYTHON_BIN "$WEKWS_ROOT/wekws/bin/compute_det_ctc.py" \
  --test_data "$EXAMPLE_DATA/test/data.list" \
  --dict "$EXAMPLE_DICT" \
  --keywords '\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee' \
  --score_file "$RESULT_DIR/score.txt" \
  --stats_dir "$RESULT_DIR/stats" \
  --det_curve_path "$RESULT_DIR/stats/det.png" \
  --step 0.001 \
  --window_shift 50
```

输出文件：

```text
$RESULT_DIR/stats/stats.嗨_小_问.txt
$RESULT_DIR/stats/stats.你_好_问_问.txt
$RESULT_DIR/stats/det.png
```

每个 stats 文件每行三列：

```text
threshold  false_alarm_per_hour  FRR
```

含义：

- `FRR`：包含该关键词的正样本中，score 低于阈值的比例；
- `false_alarm_per_hour`：不包含该关键词的负样本中，score 不低于阈值的次数 / 负样本总时长（小时）；
- `--window_shift` 当前在 `compute_det_ctc.py` 中未实际参与计算，保留默认值即可。

---

## 7. 选择工作点阈值

阈值从 0 递增到 1，FRR 单调上升，FAR 单调下降。

例如要求 FAR ≤ 0.2 次/小时：

```bash
source scripts/common/env.sh
awk '$2 <= 0.2 {print; exit}' "$RESULT_DIR/stats/stats.嗨_小_问.txt"
awk '$2 <= 0.2 {print; exit}' "$RESULT_DIR/stats/stats.你_好_问_问.txt"
```

`scripts/04_eval/eval_ctc.sh` 已把这步写进 `docs/evaluation/avg30_frr_far_report.md`。

第一条满足约束的行就是该 FAR 约束下 FRR 最低的阈值。

Android 运行时已改为每关键词独立阈值，因此不需要取两个阈值的较大值；每个词分别使用自己的 `trigger_threshold`，同一 channel 可以同时上报多个关键词事件。

### 7.1 实测结果（avg_30.pt，2026-11-26）

工作点约束取 `FAR ≤ 0.2/h`。以当前 `work/eval/avg_30_pt/score.txt` 重算后，执行上述 `awk` 得到首个满足约束的行：

```text
stats.嗨_小_问.txt
0.293 0.189454 0.039470

stats.你_好_问_问.txt
0.038 0.189813 0.032986
```

这是当前复算的理论最低 FRR 工作点 `0.293 / 0.038`。历史评测曾记录 `0.297 / 0.000`（FRR 约 `3.96%` / `3.20%`），与当前 3 位小数 `score.txt` 不一致，仅保留作追溯，不再作为选点依据。

为了给 FAR 留余量，最终部署没有取理论最低点，而是取：

```text
嗨小问   = 0.303
你好问问 = 0.039
```

最终工作点（当前 stats 对应行）：

| 关键词 | 阈值 | FAR（次/小时） | FRR |
|---|---:|---:|---:|
| 嗨小问 | 0.303 | 0.174880 | 4.0598% |
| 你好问问 | 0.039 | 0.189813 | 3.2986% |

两个阈值都满足 `FAR ≤ 0.2/h`，嗨小问余量约 `0.025/h`，你好问问余量约 `0.010/h`。早期报告中记录的 `0.174880 / 4.03%` 与 `0.175212 / 3.31%` 来自旧评测，保留在 `docs/evaluation/avg30_frr_far_report.md` 作历史记录。

不同 FAR 目标下的独立阈值参考（按当前 stats 文件重新计算）：

| FAR 目标 | 嗨小问阈值 / FRR | 你好问问阈值 / FRR |
|---:|---:|---:|
| ≤ 0.5/h | 0.154 / 2.556% | 0.000 / 3.195% |
| ≤ 0.2/h | 0.293 / 3.947% | 0.038 / 3.299% |
| ≤ 0.1/h | 0.496 / 6.343% | 0.121 / 4.689% |
| ≤ 0.05/h | 0.690 / 9.332% | 0.220 / 7.838% |

当前部署取 `≤ 0.2/h` 对应的保守工作点 `0.303 / 0.039`，而不是表中首个满足值。

---

## 8. 重新导出 ONNX 与独立阈值

`scripts/03_export/export_android_onnx.py` 直接生成独立阈值配置，不再需要导出后手工 patch。旧路径 `android/export/export_ctc_onnx.py` 仅保留兼容 wrapper。

正确步骤：

```bash
cd /workspace/wekws-my
/root/miniconda/envs/wekws/bin/python scripts/03_export/export_android_onnx.py \
  --keyword_thresholds '嗨小问=0.303,你好问问=0.039'
```

生成的 `android/app/src/main/assets/config.json` 中，每个关键词都带自己的阈值：

```json
{
  "keywords": [
    {
      "text": "嗨小问",
      "chars": ["嗨", "小", "问"],
      "tokens": [1302, 1462, 2494],
      "trigger_threshold": 0.303
    },
    {
      "text": "你好问问",
      "chars": ["你", "好", "问", "问"],
      "tokens": [1027, 1401, 2494, 2494],
      "trigger_threshold": 0.039
    }
  ]
}
```

`ctc` 段不再包含 `trigger_threshold`；缺少任一关键词阈值时导出脚本会直接失败。

导出脚本默认输出：

```text
android/app/src/main/assets/kws.onnx
android/app/src/main/assets/config.json
android/app/src/main/assets/keywords.txt
android/app/src/main/assets/id_to_token.json
```

注意：

1. 导出脚本默认 checkpoint 已固定为 `work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`，更新 epoch 不会自动替换部署资产；
2. 必须通过 `--keyword_thresholds` 显式传入所有关键词阈值，不提供默认值；
3. Android ONNX 输出是 posterior，不是 logits，Native 侧不再做 softmax；
4. score 定义与 `score_ctc.py` 一致，为 `sqrt(∏ token_prob)`；
5. 阈值只过滤事件，不改变 score、token_probs、逐帧 top-k 和 `score_file` 格式。

---

## 9. 结果记录模板

建议把评测结果记录到：

```text
docs/evaluation/avg30_frr_far_report.md
```

模板：

| 模型 | 关键词 | FAR 目标（次/小时） | 阈值 | FRR |
|---|---|---:|---:|---:|
| avg_30.pt | 嗨小问 | 0.2 | 0.303 | 4.0598% |
| avg_30.pt | 你好问问 | 0.2 | 0.039 | 3.2986% |

同时记录：

- 生成 `avg_30.pt` 时选中的 epoch 列表；
- test 集 wav 数量与负样本总时长；
- `det.png` 路径。

---

## 10. 常见问题

### 10.1 `FileNotFoundError: data/train/global_cmvn`

原因：直接拿训练产物中的 `config.yaml` 做评测 config，CMVN 是相对路径，而 cwd 不是旧 recipe 目录。

处理：使用 `scripts/04_eval/eval_ctc.sh` 生成的 `$RESULT_DIR/config.yaml`（绝对路径），或先用 `scripts/common/make_abs_cmvn_config.py` 派生 config。不要依赖 cwd。

### 10.2 关键词变成乱码

原因：`--keywords` 传了中文字面量。

处理：使用 ASCII unicode escape：

```bash
'\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee'
```

### 10.3 `average_model.py` 报 assert 失败

原因：可用 checkpoint/yaml 少于 `--num 30`。

处理：使用实际可用数量 N 生成 `avg_N.pt`，或直接用 `final.pt` / 单 checkpoint 评测。

### 10.4 GPU OOM

原因：训练与评测共用一块 GPU。

处理：

- 等训练结束再评测；或
- 将 `--batch_size` 降到 16/32。

### 10.5 阈值调到 0.001 以下没有意义

原因：`score_ctc.py` 写 score 时只保留 3 位小数。

处理：保持 `--step 0.001`。

---

## 11. 迁移后路径对照

命令入口：

```text
scripts/04_eval/eval_ctc.sh
scripts/03_export/export_android_onnx.py
scripts/05_deploy/compare_scores.sh
```

路径对应关系：

| 旧路径 | 当前路径 |
|---|---|
| `wekws/examples/hi_xiaowen/s0/exp/ds_tcn_ctc` | `work/exp/hi_xiaowen/ds_tcn_ctc` |
| `wekws/examples/hi_xiaowen/s0/data` | `work/data/hi_xiaowen` |
| `wekws/examples/hi_xiaowen/s0/dict` | `work/dict/hi_xiaowen` |
| `android/export/export_ctc_onnx.py` | `scripts/03_export/export_android_onnx.py`（旧路径保留兼容 wrapper） |
