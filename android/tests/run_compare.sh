#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_host"
WAV="${1:-$ROOT/tests/data/ch0.wav}"
CHECKPOINT="${CHECKPOINT:-$ROOT/../wekws/examples/hi_xiaowen/s0/exp/ds_tcn_ctc/15.pt}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        PYTHON_BIN="/root/miniconda/envs/wekws/bin/python"
    fi
fi

NAME="$(basename "$WAV" .wav)"
OUT="$ROOT/tests/out/$NAME"

"$PYTHON_BIN" "$ROOT/tests/onnx_reference.py" \
    --wav "$WAV" \
    --checkpoint "$CHECKPOINT" \
    --out_dir "$OUT"

T="$("$PYTHON_BIN" - <<PY
import numpy as np
print(np.load("$OUT/probs.npy").shape[0])
PY
)"

echo "Native CTC decoder on ONNX posteriors:"
"$BUILD/ctc_decoder_cli" \
    "$ROOT/app/src/main/assets/config.json" \
    "$OUT/probs.bin" "$T"

echo "Native streaming engine (fbank -> ONNX -> CTC):"
LD_LIBRARY_PATH="$BUILD" "$BUILD/kws_host_cli" \
    "$ROOT/app/src/main/assets/kws.onnx" \
    "$ROOT/app/src/main/assets/config.json" \
    "$WAV"
