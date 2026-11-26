#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

MANIFEST="$WORKSPACE_ROOT/config/pretrained/ds_tcn_ctc.env"
URL_ARG=""
DOWNLOAD_ONLY=0
FORCE=0

usage() {
    cat <<'EOF'
Usage: download_pretrained.sh [--url <URL>] [--download-only] [--force]

  --url <URL>       覆盖 config/pretrained/ds_tcn_ctc.env 中的 PRETRAINED_URL
  --download-only   只下载并校验归档，不解包、不落入 work/exp 或 work/eval
  --force           允许覆盖 work/exp 与 work/eval 中已有的同名文件
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url)
            [[ $# -ge 2 ]] || kws_die "--url requires a value"
            URL_ARG="$2"
            shift 2
            ;;
        --download-only)
            DOWNLOAD_ONLY=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            kws_die "unknown argument: $1"
            ;;
    esac
done

kws_require_file "$MANIFEST"
# shellcheck source=/dev/null
source "$MANIFEST"

URL="${URL_ARG:-${PRETRAINED_URL:-}}"
[[ -n "$URL" ]] || kws_die "no download URL: set PRETRAINED_URL in $MANIFEST or pass --url <URL>"

ARCHIVE_NAME="${PRETRAINED_ARCHIVE:-ds_tcn_ctc.tar.gz}"
EXPECTED_SHA="${PRETRAINED_SHA256:-}"
EXPECTED_AVG30="${PRETRAINED_AVG30_SHA256:-}"
[[ -n "$EXPECTED_SHA" ]] || kws_die "PRETRAINED_SHA256 missing in $MANIFEST"
[[ -n "$EXPECTED_AVG30" ]] || kws_die "PRETRAINED_AVG30_SHA256 missing in $MANIFEST"

sha_of() {
    sha256sum "$1" | awk '{print $1}'
}

mkdir -p "$PRETRAINED_DIR"
ARCHIVE="$PRETRAINED_DIR/$ARCHIVE_NAME"

if [[ -f "$ARCHIVE" ]] && [[ "$(sha_of "$ARCHIVE")" == "$EXPECTED_SHA" ]]; then
    echo "skip: $ARCHIVE already verified"
else
    command -v curl >/dev/null || kws_die "curl not found"
    TMP="$PRETRAINED_DIR/$ARCHIVE_NAME.part.$$"
    echo "downloading $URL -> $TMP"
    curl -fL --retry 3 --retry-delay 2 -o "$TMP" "$URL"
    if [[ "$(sha_of "$TMP")" != "$EXPECTED_SHA" ]]; then
        rm -f "$TMP"
        kws_die "downloaded archive sha256 mismatch: $TMP"
    fi
    mv -f "$TMP" "$ARCHIVE"
    echo "archive verified: $ARCHIVE"
fi

if [[ "$DOWNLOAD_ONLY" == "1" ]]; then
    echo "download-only: $ARCHIVE"
    exit 0
fi

STAGE="$PRETRAINED_DIR/ds_tcn_ctc"
rm -rf "$STAGE"
mkdir -p "$STAGE"
tar -xzf "$ARCHIVE" -C "$STAGE" --strip-components=1

kws_require_file "$STAGE/config.yaml"
kws_require_file "$STAGE/avg_30.pt"
kws_require_file "$STAGE/init.pt"
kws_require_file "$STAGE/final.pt"
[[ -d "$STAGE/test_avg_30_pt" ]] || kws_die "eval outputs missing in archive"

EPOCHS="$(find "$STAGE" -maxdepth 1 -type f -name '*.pt' -printf '%f\n' | sed 's/\.pt$//' | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
EXPECTED_EPOCHS="$(seq 0 79 | tr '\n' ' ')"
[[ "$EPOCHS" == "$EXPECTED_EPOCHS" ]] || kws_die "epoch sequence mismatch in archive"
[[ "$(sha_of "$STAGE/avg_30.pt")" == "$EXPECTED_AVG30" ]] || kws_die "avg_30.pt sha256 mismatch in archive"

copy_item() {
    local src="$1" dst="$2"
    mkdir -p "$dst"
    if [[ "$FORCE" == "1" ]]; then
        cp -a "$src" "$dst/"
    else
        cp -a --update=none "$src" "$dst/"
    fi
}

while IFS= read -r -d '' item; do
    copy_item "$item" "$EXAMPLE_EXP"
done < <(find "$STAGE" -mindepth 1 -maxdepth 1 ! -name 'test_avg_30_pt' -print0)
copy_item "$STAGE/test_avg_30_pt/." "$RESULT_DIR"

while IFS= read -r -d '' f; do
    rel="${f#"$STAGE/"}"
    cmp -s "$f" "$EXAMPLE_EXP/$rel" || kws_die "$EXAMPLE_EXP/$rel differs from archive; rerun with --force to replace"
done < <(find "$STAGE" -mindepth 1 -type f ! -path "$STAGE/test_avg_30_pt/*" -print0)

while IFS= read -r -d '' f; do
    rel="${f#"$STAGE/test_avg_30_pt/"}"
    cmp -s "$f" "$RESULT_DIR/$rel" || kws_die "$RESULT_DIR/$rel differs from archive; rerun with --force to replace"
done < <(find "$STAGE/test_avg_30_pt" -type f -print0)

echo "pretrained artifacts ready in $EXAMPLE_EXP"
echo "eval outputs ready in $RESULT_DIR"
