#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/app/src/main/cpp"
BUILD="$ROOT/build_host"
mkdir -p "$BUILD"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        PYTHON_BIN="/root/miniconda/envs/wekws/bin/python"
    fi
fi

if [[ -z "${ORT_INC:-}" ]]; then
    if [[ -f "$ROOT/third_party/onnxruntime/include/onnxruntime_cxx_api.h" ]]; then
        ORT_INC="$ROOT/third_party/onnxruntime/include"
    elif [[ -f /tmp/opencode/ortcheck/candidates/h-1.20.0/headers/onnxruntime_cxx_api.h ]]; then
        # Local fallback used in this workspace; still download with
        # export/download_ort_mobile.sh for a portable build.
        ORT_INC="/tmp/opencode/ortcheck/candidates/h-1.20.0/headers"
    else
        echo "ORT headers not found; run: bash $ROOT/export/download_ort_mobile.sh" >&2
        exit 1
    fi
fi

ORT_LIB_DIR="$($PYTHON_BIN - <<'PY'
import pathlib
import onnxruntime
print(pathlib.Path(onnxruntime.__file__).parent / "capi")
PY
)"
ORT_LIB="$(find "$ORT_LIB_DIR" -maxdepth 1 -name 'libonnxruntime.so*' -type f | head -1)"
if [[ -z "$ORT_LIB" ]]; then
    echo "libonnxruntime.so not found in $ORT_LIB_DIR" >&2
    exit 1
fi
ORT_LIB_NAME="$(basename "$ORT_LIB")"
ln -sf "$ORT_LIB" "$BUILD/libonnxruntime.so.1"

COMMON_ENGINE=(
    "$SRC/engine/json.cc"
    "$SRC/engine/runtime_config.cc"
    "$SRC/engine/ctc_beam_search.cc"
    "$SRC/engine/frontend/fft.cc"
    "$SRC/engine/frontend/fbank.cc"
    "$SRC/engine/frontend/feature_pipeline.cc"
)

echo "Building kws_host_cli"
g++ -std=c++14 -O2 -pthread -I"$SRC" -I"$ORT_INC" \
    "${COMMON_ENGINE[@]}" \
    "$SRC/engine/kws_engine.cc" \
    "$SRC/engine/wav_reader.cc" \
    "$SRC/engine/kws_host_cli.cc" \
    -L"$ORT_LIB_DIR" -l:"$ORT_LIB_NAME" \
    -Wl,-rpath,"$ORT_LIB_DIR" \
    -o "$BUILD/kws_host_cli"

echo "Building ctc_decoder_cli"
g++ -std=c++14 -O2 -pthread -I"$SRC" \
    "$SRC/engine/json.cc" \
    "$SRC/engine/runtime_config.cc" \
    "$SRC/engine/ctc_beam_search.cc" \
    "$SRC/engine/ctc_decoder_cli.cc" \
    -o "$BUILD/ctc_decoder_cli"

echo "Building fbank_dump_cli"
g++ -std=c++14 -O2 -pthread -I"$SRC" \
    "$SRC/engine/json.cc" \
    "$SRC/engine/runtime_config.cc" \
    "$SRC/engine/frontend/fft.cc" \
    "$SRC/engine/frontend/fbank.cc" \
    "$SRC/engine/frontend/feature_pipeline.cc" \
    "$SRC/engine/wav_reader.cc" \
    "$SRC/engine/fbank_dump_cli.cc" \
    -o "$BUILD/fbank_dump_cli"

echo "Host binaries are in $BUILD"
echo "Run with: LD_LIBRARY_PATH=$BUILD $BUILD/kws_host_cli $ROOT/app/src/main/assets/kws.onnx $ROOT/app/src/main/assets/config.json $ROOT/tests/data/ch0.wav"
