# avg_30.pt FRR/FAR 评测结果

> 生成时间：2026-11-27T13:16（cooldown/reset 生效后完整重跑）
> checkpoint：`work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`（sha256 `fa02f745cb48fdb961893981e5bd7e8836738e47569ea7997a723ed09e12149c`）
> score 文件：`work/eval/avg_30_pt/score.txt`（3 位小数，73459 行）
> DET 参数：`--step 0.001`、`--window_shift 50`
> stats：`work/eval/avg_30_pt/stats/stats.嗨_小_问.txt`、`stats.你_好_问_问.txt`
> DET 图：`work/eval/avg_30_pt/stats/det.png`
> 对齐说明：`docs/evaluation/streaming_alignment_report.md`

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
|---|---|---|---:|---:|---:|
| avg_30.pt | 嗨小问 | 0.2 | 0.303 | 0.174880 | 4.0598% |
| avg_30.pt | 你好问问 | 0.2 | 0.039 | 0.189813 | 3.2986% |

历史报告中记录的部署 FAR/FRR 为 `0.174880 / 4.03%`（嗨小问）与 `0.175212 / 3.31%`（你好问问），同样来自早期评测；当前 stats 对应行分别是 `0.174880 / 4.0598%` 与 `0.189813 / 3.2986%`。以当前 stats 为准，部署阈值仍为 `0.303 / 0.039`：嗨小问保留约 `0.025/h` FAR 余量，你好问问保留约 `0.010/h` 余量，均满足 `FAR ≤ 0.2/h`。per-keyword 阈值彼此独立，同一 channel 可以同时上报两个关键词事件。

## cooldown/reset 生效后的事件次数

离线 FRR/FAR 由 `score_ctc.py` 的整段 beam 生成 **不受运行时 cooldown/reset 影响**；触发控制只改变流式事件次数、上报时机和 `currentEvents()` 观察到的分数。因此本次重跑不修改部署阈值，也不需要为触发控制重新选点。

下表用 `onnx_kws_core.py` 的 100 ms 分块流式会话复测；`ch0_twice.wav` 由 `android/tests/data/ch0.wav` 原样拼接两遍生成，两个唤醒间隔约 2 秒。配置均由 `android/app/src/main/assets/config.json` 派生：

| WAV | 配置 | cooldown / reset | `triggered_events` 次数 | 去重签名（ms） | 最终 `currentEvents()` |
|---|---|---|---:|---|---|
| `ch0_twice.wav` | deployed | `1000 / decoder=true, cache=false` | 2 | `690-1290`, `2710-3310` | `0.760113 / 2710-3310` |
| `ch0_twice.wav` | baseline | `0 / decoder=false, cache=false` | 273 | `690-1290` | `0.783477 / 690-1290` |
| `ch0_twice.wav` | cd3000 | `3000 / decoder=true, cache=false` | 1 | `690-1290` | `0.783477 / 690-1290` |
| `ch0_twice.wav` | decoder_only | `0 / decoder=true, cache=false` | 2 | `690-1290`, `2710-3310` | `0.760113 / 2710-3310` |
| `ch_nihao.wav` | deployed | `1000 / decoder=true, cache=false` | 1 | `630-1070` | `0.088597 / 630-1070` |
| `ch_nihao.wav` | baseline | `0 / decoder=false, cache=false` | 84 | `630-1070`, `630-1080`, `630-1250` | `0.224240 / 630-1250` |
| `ch_nihao.wav` | cd3000 | `3000 / decoder=true, cache=false` | 1 | `630-1070` | `0.088597 / 630-1070` |
| `ch_nihao.wav` | decoder_only | `0 / decoder=true, cache=false` | 1 | `630-1070` | `0.088597 / 630-1070` |

说明：

- baseline 的 `273` / `84` 是同一关键词在 beam 持续存在期间逐帧重复上报，不代表独立唤醒次数；部署默认把上报收敛为 `2` / `1` 次。
- `cd3000` 将第二次 `嗨小问` 抑制，证明 cooldown 按 channel/关键词生效；它不影响其他关键词。
- `ch_nihao.wav` 在 deployed/decoder_only 下停在 `0.088597 / 63-107`，而 baseline 锚点为 `0.224240 / 63-125`。差异来自命中后立刻 reset decoder，改变了 `currentEvents()` 的观察时机，不是 `sqrt(∏ token_prob)` 公式或阈值变化；完整四路证据见 `docs/evaluation/streaming_alignment_report.md`。

事件次数复测方式：

```bash
source scripts/common/env.sh
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
import numpy as np
import soundfile as sf

out = Path('work/eval/streaming_controls')
out.mkdir(parents=True, exist_ok=True)
pcm, sr = sf.read('android/tests/data/ch0.wav', dtype='int16', always_2d=True)
sf.write(out / 'ch0_twice.wav', np.concatenate([pcm, pcm], axis=0),
         sr, subtype='PCM_16')
base = json.load(open('android/app/src/main/assets/config.json', encoding='utf-8'))
variants = {
    'deployed': {},
    'baseline': dict(cooldown_ms=0, reset_decoder_after_trigger=False,
                     reset_cache_after_trigger=False),
    'cd3000': dict(cooldown_ms=3000, reset_decoder_after_trigger=True,
                   reset_cache_after_trigger=False),
    'decoder_only': dict(cooldown_ms=0, reset_decoder_after_trigger=True,
                         reset_cache_after_trigger=False),
}
for name, patch in variants.items():
    cfg = json.loads(json.dumps(base))
    cfg['ctc'].update(patch)
    json.dump(cfg, open(out / f'config_{name}.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
PY

for cfg in deployed baseline cd3000 decoder_only; do
  for wav in work/eval/streaming_controls/ch0_twice.wav android/tests/data/ch_nihao.wav; do
    "$PYTHON_BIN" scripts/04_eval/onnx_kws_core.py \
      --config_json "work/eval/streaming_controls/config_$cfg.json" \
      --onnx "$PWD/android/app/src/main/assets/kws.onnx" \
      --wav "$wav" --chunk_ms 100
  done
done
```

## 复现命令

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh

# 常规入口（会重写本报告，新 checkpoint 时必须重新按 FAR ≤ 0.2/h 选点）
bash scripts/04_eval/eval_ctc.sh --checkpoint "$DEPLOY_CHECKPOINT" --result_dir "$RESULT_DIR"

# 本机重跑时，上游 score_ctc.py 默认 prefetch=100 会触发 DataLoader worker OOM，
# 因此本次显式降低 prefetch；score/stats 格式与 eval_ctc.sh 完全一致。
RESULT_DIR="$WORK/eval/avg_30_pt_streaming"
mkdir -p "$RESULT_DIR/stats"
"$PYTHON_BIN" "$WEKWS_ROOT/wekws/bin/score_ctc.py" \
  --config "$RESULT_DIR/config.yaml" \
  --dict "$EXAMPLE_DICT" \
  --test_data "$EXAMPLE_DATA/test/data.list" \
  --gpu 0 --checkpoint "$DEPLOY_CHECKPOINT" \
  --batch_size 64 --num_workers 4 --prefetch 4 \
  --score_file "$RESULT_DIR/score.txt" --keywords "$KEYWORDS_ESCAPED"
"$PYTHON_BIN" "$WEKWS_ROOT/wekws/bin/compute_det_ctc.py" \
  --dict "$EXAMPLE_DICT" --keywords "$KEYWORDS_ESCAPED" \
  --test_data "$EXAMPLE_DATA/test/data.list" \
  --window_shift 50 --step 0.001 \
  --score_file "$RESULT_DIR/score.txt" \
  --stats_dir "$RESULT_DIR/stats" \
  --det_curve_path "$RESULT_DIR/stats/det.png"

awk '$2 <= 0.2 {print; exit}' "$RESULT_DIR/stats/stats.嗨_小_问.txt"
awk '$2 <= 0.2 {print; exit}' "$RESULT_DIR/stats/stats.你_好_问_问.txt"
```

评测脚本会派生带绝对 CMVN 路径的 `$RESULT_DIR/config.yaml`，不依赖旧 recipe cwd。`score_ctc.py` 无 `--cmvn_file` 参数，手工调用时必须使用该派生 config 或固定到可解析的相对路径 cwd。关键词必须传 ASCII escape：`'\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee'`。
