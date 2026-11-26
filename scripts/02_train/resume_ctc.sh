#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

CONFIG="$CONFIG_DIR/ds_tcn_ctc_resume.yaml"
CHECKPOINT=""
MODEL_DIR="$WORK/exp/hi_xiaowen/ds_tcn_ctc_resume"
TENSORBOARD_DIR="$WORK/tensorboard/hi_xiaowen/ds_tcn_ctc_resume"
GPUS="${GPUS:-0}"
SEED="${SEED:-666}"
NUM_WORKERS="${NUM_WORKERS:-8}"

usage() {
    cat <<EOF
Usage: $0 --checkpoint FILE [--model_dir DIR] [--tensorboard_dir DIR]
          [--gpus IDS] [--num_workers N]

Resume training uses $CONFIG and writes to an independent output directory.
The checkpoint is required and must not be init.pt.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --model_dir) MODEL_DIR="$2"; shift 2 ;;
        --tensorboard_dir) TENSORBOARD_DIR="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) kws_die "unknown argument: $1" ;;
    esac
done

[[ -n "$CHECKPOINT" ]] || kws_die "--checkpoint is required for resume"
[[ "$(basename "$CHECKPOINT")" != "init.pt" ]] \
    || kws_die "init.pt is not a resumable checkpoint; pass a trained .pt file"

kws_env_ready
kws_require_file "$CONFIG"
kws_require_file "$EXAMPLE_DICT/dict.txt"
kws_require_file "$EXAMPLE_DATA/train/data.list"
kws_require_file "$EXAMPLE_DATA/dev/data.list"
kws_require_file "$EXAMPLE_DATA/train/global_cmvn"
kws_require_file "$CHECKPOINT"

mkdir -p "$MODEL_DIR" "$TENSORBOARD_DIR"

num_gpus="$(awk -F ',' '{print NF}' <<<"$GPUS")"
CMD=(
    "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$num_gpus"
    "$WEKWS_ROOT/wekws/bin/train.py"
    --gpus "$GPUS"
    --config "$CONFIG"
    --dict "$EXAMPLE_DICT"
    --train_data "$EXAMPLE_DATA/train/data.list"
    --cv_data "$EXAMPLE_DATA/dev/data.list"
    --model_dir "$MODEL_DIR"
    --tensorboard_dir "$TENSORBOARD_DIR"
    --num_workers "$NUM_WORKERS"
    --num_keywords 2599
    --min_duration 50
    --seed "$SEED"
    --cmvn_file "$EXAMPLE_DATA/train/global_cmvn"
    --norm_var
    --checkpoint "$CHECKPOINT"
)

echo "resume checkpoint: $CHECKPOINT"
echo "resume config:     $CONFIG"
echo "resume model_dir:  $MODEL_DIR"
kws_run "${CMD[@]}"
