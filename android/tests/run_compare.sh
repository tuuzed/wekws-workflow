#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_host"
CHECKPOINT="${CHECKPOINT:-$ROOT/../work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        PYTHON_BIN="/root/miniconda/envs/wekws/bin/python"
    fi
fi

WAV=""
ONNX_ONLY=()
usage() {
    cat <<EOF
Usage: $0 [--onnx-only] [wav]

  --onnx-only  Skip train CMVN and PyTorch checkpoint comparison. This is the
               default verification path when work/data/.../global_cmvn is absent.
  wav          Input WAV (default: $ROOT/tests/data/ch0.wav)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx-only) ONNX_ONLY+=("--onnx_only"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) WAV="$1"; shift ;;
    esac
done
WAV="${WAV:-$ROOT/tests/data/ch0.wav}"

NAME="$(basename "$WAV" .wav)"
OUT="$ROOT/tests/out/$NAME"

"$PYTHON_BIN" "$ROOT/tests/onnx_reference.py" \
    --wav "$WAV" \
    --checkpoint "$CHECKPOINT" \
    --out_dir "$OUT" \
    "${ONNX_ONLY[@]}"

T="$("$PYTHON_BIN" - <<PY
import numpy as np
print(np.load("$OUT/probs.npy").shape[0])
PY
)"

echo "Native posterior decoder on ONNX posteriors:"
"$BUILD/ctc_decoder_cli" \
    "$ROOT/app/src/main/assets/config.json" \
    "$OUT/probs.bin" "$T"

echo "Native streaming engine (fbank -> ONNX -> CTC):"
LD_LIBRARY_PATH="$BUILD" "$BUILD/kws_host_cli" \
    "$ROOT/app/src/main/assets/kws.onnx" \
    "$ROOT/app/src/main/assets/config.json" \
    "$WAV"
