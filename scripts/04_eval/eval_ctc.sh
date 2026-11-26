#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

CHECKPOINT="$DEPLOY_CHECKPOINT"
RESULT_DIR="${RESULT_DIR:-$EVAL_ROOT/avg_30_pt}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FAR_TARGET="${FAR_TARGET:-0.2}"

usage() {
    cat <<EOF
Usage: $0 [--checkpoint FILE] [--result_dir DIR] [--gpu ID]
          [--batch_size N] [--num_workers N] [--far_target FAR]

Defaults:
  checkpoint  $CHECKPOINT (falls back to avg_N.pt / final.pt)
  result_dir  $RESULT_DIR
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --result_dir) RESULT_DIR="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --far_target) FAR_TARGET="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) kws_die "unknown argument: $1" ;;
    esac
done

kws_env_ready
kws_require_dir "$EXAMPLE_EXP"
kws_require_file "$EXAMPLE_DICT/dict.txt"
kws_require_file "$EXAMPLE_DATA/test/data.list"

# average_model.py --val_best asserts num == len(path_list), so fall back to
# the actual number of available checkpoint/yaml pairs when fewer than 30 are
# present. Never silently switch the deployment checkpoint to a newer epoch.
if [[ ! -f "$CHECKPOINT" ]]; then
    count="$(find "$EXAMPLE_EXP" -maxdepth 1 -type f -name '*.yaml' \
        ! -name 'config.yaml' ! -name 'init.yaml' | wc -l)"
    if (( count > 0 )); then
        num="$(( count < 30 ? count : 30 ))"
        echo "checkpoint $CHECKPOINT missing; averaging $num best epochs"
        "$PYTHON_BIN" "$WEKWS_ROOT/wekws/bin/average_model.py" \
            --dst_model "$EXAMPLE_EXP/avg_${num}.pt" \
            --src_path "$EXAMPLE_EXP" \
            --num "$num" \
            --val_best
        CHECKPOINT="$EXAMPLE_EXP/avg_${num}.pt"
    elif [[ -f "$EXAMPLE_EXP/final.pt" ]]; then
        echo "checkpoint missing; falling back to final.pt"
        CHECKPOINT="$EXAMPLE_EXP/final.pt"
    else
        kws_die "no checkpoint or final.pt available in $EXAMPLE_EXP"
    fi
fi
kws_require_file "$CHECKPOINT"

mkdir -p "$RESULT_DIR/stats"

# The trained config.yaml stores a relative CMVN path (data/train/global_cmvn).
# Rewrite it into an absolute path for evaluation instead of depending on the
# old recipe cwd. The training artifact itself is left untouched.
EVAL_CONFIG="$RESULT_DIR/config.yaml"
"$PYTHON_BIN" - "$EXAMPLE_EXP/config.yaml" "$EVAL_CONFIG" "$EXAMPLE_DATA/train/global_cmvn" <<'PY'
import sys
import yaml

src, dst, cmvn = sys.argv[1:4]
with open(src, encoding='utf-8') as fin:
    config = yaml.safe_load(fin)
config.setdefault('model', {}).setdefault('cmvn', {})['cmvn_file'] = cmvn
with open(dst, 'w', encoding='utf-8') as fout:
    yaml.safe_dump(config, fout, sort_keys=False, allow_unicode=True)
print(f'eval config with absolute cmvn: {dst}')
PY

SCORE_FILE="$RESULT_DIR/score.txt"
CMD=(
    "$PYTHON_BIN" "$WEKWS_ROOT/wekws/bin/score_ctc.py"
    --config "$EVAL_CONFIG"
    --dict "$EXAMPLE_DICT"
    --test_data "$EXAMPLE_DATA/test/data.list"
    --gpu "$GPU"
    --checkpoint "$CHECKPOINT"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --score_file "$SCORE_FILE"
    --keywords "$KEYWORDS_ESCAPED"
)
echo "score checkpoint: $CHECKPOINT"
kws_run "${CMD[@]}"

CMD=(
    "$PYTHON_BIN" "$WEKWS_ROOT/wekws/bin/compute_det_ctc.py"
    --dict "$EXAMPLE_DICT"
    --keywords "$KEYWORDS_ESCAPED"
    --test_data "$EXAMPLE_DATA/test/data.list"
    --window_shift 50
    --step 0.001
    --score_file "$SCORE_FILE"
    --stats_dir "$RESULT_DIR/stats"
    --det_curve_path "$RESULT_DIR/stats/det.png"
)
kws_run "${CMD[@]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] skipping operating-point report"
    exit 0
fi

REPORT="docs/evaluation/avg30_frr_far_report.md"
mkdir -p "$(dirname "$REPORT")"
{
    echo "# avg_30.pt FRR/FAR 评测结果"
    echo
    echo "生成时间：$(date -Is)"
    echo "checkpoint：$CHECKPOINT"
    echo "score 文件：$SCORE_FILE"
    echo "stats 目录：$RESULT_DIR/stats"
    echo "FAR 目标：$FAR_TARGET 次/小时"
    echo
    echo "## 理论最低工作点（awk '\$2 <= $FAR_TARGET {print; exit}'）"
    echo
    echo "| stats 文件 | threshold | FAR/h | FRR |"
    echo "|---|---:|---:|---:|"
    for pair in "嗨_小_问:嗨小问" "你_好_问_问:你好问问"; do
        name="${pair%%:*}"
        word="${pair##*:}"
        line="$(awk -v target="$FAR_TARGET" '$2 <= target {print; exit}' "$RESULT_DIR/stats/stats.$name.txt")"
        read -r threshold far frr <<<"$line"
        printf '| %s | %s | %s | %s |\n' "$word" "$threshold" "$far" "$frr"
    done
    echo
    echo "## 部署工作点（留余量）"
    echo
    echo "| 关键词 | 阈值 |"
    echo "|---|---:|"
    echo "| 嗨小问 | 0.303 |"
    echo "| 你好问问 | 0.039 |"
} > "$REPORT"

echo "report written to $REPORT"
