#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

SRC="$WEKWS_ROOT/examples/hi_xiaowen/s0"
COMMIT_OLD=0
if [[ "${1:-}" == "--commit" ]]; then
    COMMIT_OLD=1
fi

if pgrep -fa 'torchrun|wekws/bin/train.py|wekws.bin.score_ctc|wekws.bin.compute_det_ctc|export_onnx|export_jit' 2>/dev/null \
    | grep -v 'migrate_recipe_artifacts.sh' | grep -q .; then
    kws_die "active upstream writer detected (train/score/det/export); wait for it to finish before migrating artifacts"
fi

migrate_one() {
    local src="$1"
    local dst="$2"
    kws_require_dir "$src"
    mkdir -p "$dst"

    local src_files dst_files src_bytes dst_bytes
    src_files="$(find "$src" -type f | wc -l)"
    src_bytes="$(du -sb "$src" | cut -f1)"

    cp -a "$src/." "$dst/"

    dst_files="$(find "$dst" -type f | wc -l)"
    dst_bytes="$(du -sb "$dst" | cut -f1)"
    [[ "$src_files" == "$dst_files" ]] || kws_die "file count mismatch: $src -> $dst"
    [[ "$src_bytes" == "$dst_bytes" ]] || kws_die "byte size mismatch: $src -> $dst"
    printf 'migrated %s -> %s (%s files, %s bytes)\n' "$src" "$dst" "$dst_files" "$dst_bytes"
}

EXP_PARENT="$(dirname "$EXAMPLE_EXP")"
TB_PARENT="$(dirname "$EXAMPLE_TB")"

if [[ "${VERIFY_ONLY:-0}" != "1" ]]; then
    migrate_one "$SRC/data" "$EXAMPLE_DATA"
    migrate_one "$SRC/dict" "$EXAMPLE_DICT"
    migrate_one "$SRC/exp" "$EXP_PARENT"
    migrate_one "$SRC/tensorboard" "$TB_PARENT"
fi

EPOCH_SRC="$(find "$SRC/exp/ds_tcn_ctc" -maxdepth 1 -type f -name '*.pt' -printf '%f\n' | sed 's/\.pt$//' | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
EPOCH_DST="$(find "$EXAMPLE_EXP" -maxdepth 1 -type f -name '*.pt' -printf '%f\n' | sed 's/\.pt$//' | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
[[ "$EPOCH_SRC" == "$EPOCH_DST" ]] || kws_die "epoch sequence mismatch"

sha256sum "$SRC/exp/ds_tcn_ctc/avg_30.pt" "$EXAMPLE_EXP/avg_30.pt"

if [[ "$COMMIT_OLD" == "1" ]]; then
    rm -rf "$SRC/data" "$SRC/dict" "$SRC/exp" "$SRC/tensorboard"
    echo "old recipe artifact directories removed"
else
    echo "old recipe artifact directories kept for rollback; rerun with --commit after Phase 4 acceptance"
fi
