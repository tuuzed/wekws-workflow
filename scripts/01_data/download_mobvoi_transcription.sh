#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

REPO_URL="${MOBVOI_TRANSCRIPTION_URL:-https://www.modelscope.cn/datasets/thuduj12/mobvoi_kws_transcription.git}"
mkdir -p "$DOWNLOADS"

if [[ -d "$MOBVOI_TRANSCRIPTION_DIR/.git" ]]; then
    echo "skip: $MOBVOI_TRANSCRIPTION_DIR already exists"
elif [[ -e "$MOBVOI_TRANSCRIPTION_DIR" ]]; then
    kws_die "$MOBVOI_TRANSCRIPTION_DIR exists but is not a mobvoi_kws_transcription git checkout"
else
    echo "cloning $REPO_URL into $MOBVOI_TRANSCRIPTION_DIR"
    git clone "$REPO_URL" "$MOBVOI_TRANSCRIPTION_DIR"
fi

kws_require_file "$MOBVOI_TRANSCRIPTION_DIR/23.pt"
kws_require_file "$MOBVOI_TRANSCRIPTION_DIR/tokens.txt"
echo "mobvoi transcription ready in $MOBVOI_TRANSCRIPTION_DIR"
