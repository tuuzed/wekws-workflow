#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

WAV_DIR="$MOBVOI_DIR/mobvoi_hotword_dataset"
RESOURCE_DIR="$MOBVOI_DIR/mobvoi_hotword_dataset_resources"
DICT="$EXAMPLE_DICT/dict.txt"

kws_require_dir "$WAV_DIR"
kws_require_dir "$RESOURCE_DIR"
kws_require_file "$DICT"
kws_require_file "$CONFIG_DIR/ds_tcn_ctc.yaml"

# Upstream tools/ scripts resolve `tools/parse_options.sh` and `tools/*.py`
# relative to the current directory. Keep cwd at the upstream root, but pass
# every input/output as an absolute path under work/ so nothing is written
# back into the read-only checkout.
cd "$WEKWS_ROOT"

PREP="$WEKWS_ROOT/examples/hi_xiaowen/s0/local/prepare_data.py"

for split in train dev test; do
    out="$EXAMPLE_DATA/$split"
    mkdir -p "$out"

    for prefix in p n; do
        tmp="$EXAMPLE_DATA/${prefix}_$split"
        mkdir -p "$tmp"
        json="$RESOURCE_DIR/${prefix}_$split.json"
        kws_require_file "$json"
        "$PYTHON_BIN" "$PREP" "$WAV_DIR" "$json" "$DICT" "$tmp"
    done

    cat "$EXAMPLE_DATA/p_$split/wav.scp" "$EXAMPLE_DATA/n_$split/wav.scp" > "$out/wav.scp"
    cat "$EXAMPLE_DATA/p_$split/text" "$EXAMPLE_DATA/n_$split/text" > "$out/text"
    rm -rf "$EXAMPLE_DATA/p_$split" "$EXAMPLE_DATA/n_$split"

    if [[ -f "$MOBVOI_TRANSCRIPTION_DIR/$split.text" ]]; then
        if [[ -f "$out/text" && ! -f "$out/text.label" ]]; then
            mv "$out/text" "$out/text.label"
        fi
        cp "$MOBVOI_TRANSCRIPTION_DIR/$split.text" "$out/text"
    fi
done

# CMVN uses absolute paths for both the scp and the output; the config is the
# outer project copy so changing the upstream yaml cannot affect this step.
"$PYTHON_BIN" "$WEKWS_ROOT/tools/compute_cmvn_stats.py" \
    --num_workers 16 \
    --train_config "$CONFIG_DIR/ds_tcn_ctc.yaml" \
    --in_scp "$EXAMPLE_DATA/train/wav.scp" \
    --out_cmvn "$EXAMPLE_DATA/train/global_cmvn"

for split in train dev test; do
    d="$EXAMPLE_DATA/$split"
    bash "$WEKWS_ROOT/tools/wav_to_duration.sh" --nj 8 "$d/wav.scp" "$d/wav.dur"
    "$PYTHON_BIN" "$WEKWS_ROOT/tools/make_list.py" \
        "$d/wav.scp" "$d/text" "$d/wav.dur" "$d/data.list"
done

echo "mobvoi data prepared in $EXAMPLE_DATA"
