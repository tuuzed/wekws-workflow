#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

DATA_URL="${DATA_URL:-http://www.openslr.org/resources/87}"
DATASET=mobvoi_hotword_dataset.tgz
RESOURCES=mobvoi_hotword_dataset_resources.tgz

mkdir -p "$MOBVOI_DIR"

need_download() {
    local archive="$1"
    local extracted="${archive%.tgz}"
    if [[ -f "$MOBVOI_DIR/$archive" || -d "$MOBVOI_DIR/$extracted" ]]; then
        return 1
    fi
    return 0
}

download_one() {
    local archive="$1"
    if ! need_download "$archive"; then
        echo "skip $archive: already present in $MOBVOI_DIR"
        return 0
    fi
    command -v wget >/dev/null 2>&1 || kws_die "wget not found; install wget first"
    echo "downloading $archive from $DATA_URL/$archive"
    wget --no-check-certificate -O "$MOBVOI_DIR/$archive" "$DATA_URL/$archive" \
        || kws_die "failed to download $archive"
}

extract_one() {
    local archive="$1"
    local extracted="${archive%.tgz}"
    if [[ -d "$MOBVOI_DIR/$extracted" ]]; then
        echo "skip extract $archive: $extracted already exists"
        return 0
    fi
    kws_require_file "$MOBVOI_DIR/$archive"
    echo "extracting $archive"
    tar -xzf "$MOBVOI_DIR/$archive" -C "$MOBVOI_DIR"
}

download_one "$DATASET"
download_one "$RESOURCES"
extract_one "$DATASET"
extract_one "$RESOURCES"

echo "mobvoi data ready in $MOBVOI_DIR"
