#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

CHECKPOINT="$DEPLOY_CHECKPOINT"
MODEL="$EXPORT_DIR/avg_30.zip"

usage() {
    cat <<EOF
Usage: $0 [--checkpoint FILE] [--jit_model FILE]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --jit_model) MODEL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) kws_die "unknown argument: $1" ;;
    esac
done

kws_env_ready
kws_require_file "$CHECKPOINT"
mkdir -p "$EXPORT_DIR"

CONFIG="$EXPORT_DIR/config_abs_cmvn.yaml"
"$PYTHON_BIN" "$WORKSPACE_ROOT/scripts/common/make_abs_cmvn_config.py" \
    "$EXAMPLE_EXP/config.yaml" "$CONFIG" "$EXAMPLE_DATA/train/global_cmvn"

CMD=(
    "$PYTHON_BIN" "$WEKWS_ROOT/wekws/bin/export_jit.py"
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --jit_model "$MODEL"
)
kws_run "${CMD[@]}"
echo "generic JIT: $MODEL"
