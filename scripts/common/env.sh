#!/usr/bin/env bash
# shellcheck shell=bash
# Single source of truth for paths, Python environment and deployment constants.

if [[ -n "${BASH_VERSION:-}" ]]; then
    KWS_ENV_SOURCE="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
    KWS_ENV_SOURCE="${(%):-%x}"
else
    KWS_ENV_SOURCE="$0"
fi
KWS_ENV_SOURCE="$(cd "$(dirname "$KWS_ENV_SOURCE")" && pwd)"
WORKSPACE_ROOT="$(cd "$KWS_ENV_SOURCE/../.." && pwd)"
unset KWS_ENV_SOURCE

WEKWS_ROOT="$WORKSPACE_ROOT/wekws"
WORK="$WORKSPACE_ROOT/work"
CONFIG_DIR="$WORKSPACE_ROOT/config/hi_xiaowen"

DOWNLOADS="$WORK/downloads"
MOBVOI_DIR="$DOWNLOADS/mobvoi"
MOBVOI_TRANSCRIPTION_DIR="$DOWNLOADS/mobvoi_kws_transcription"

EXAMPLE_DATA="$WORK/data/hi_xiaowen"
EXAMPLE_DICT="$WORK/dict/hi_xiaowen"
EXAMPLE_EXP="$WORK/exp/hi_xiaowen/ds_tcn_ctc"
EXAMPLE_TB="$WORK/tensorboard/hi_xiaowen/ds_tcn_ctc"
EXPORT_DIR="$WORK/exports/hi_xiaowen"
EVAL_ROOT="$WORK/eval"
RESULT_DIR="${RESULT_DIR:-$EVAL_ROOT/avg_30_pt}"

BASE_CHECKPOINT="$MOBVOI_TRANSCRIPTION_DIR/23.pt"
DEPLOY_CHECKPOINT="$EXAMPLE_EXP/avg_30.pt"
KEYWORDS='嗨小问,你好问问'
KEYWORD_THRESHOLDS='嗨小问=0.303,你好问问=0.039'
KEYWORDS_ESCAPED='\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee'

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/envs/wekws/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda/envs/wekws/bin/torchrun}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-32}"

export WORKSPACE_ROOT WEKWS_ROOT WORK CONFIG_DIR
export DOWNLOADS MOBVOI_DIR MOBVOI_TRANSCRIPTION_DIR
export EXAMPLE_DATA EXAMPLE_DICT EXAMPLE_EXP EXAMPLE_TB EXPORT_DIR EVAL_ROOT RESULT_DIR
export BASE_CHECKPOINT DEPLOY_CHECKPOINT KEYWORDS KEYWORD_THRESHOLDS KEYWORDS_ESCAPED
export PYTHON_BIN TORCHRUN_BIN SCORE_BATCH_SIZE

export PYTHONPATH="$WEKWS_ROOT:${PYTHONPATH:-}"
export PYTHONIOENCODING=UTF-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Keep bytecode out of the read-only upstream checkout.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$WORK/pycache"

kws_die() {
    echo "ERROR: $*" >&2
    exit 1
}

kws_require_file() {
    [[ -f "$1" ]] || kws_die "required file missing: $1"
}

kws_require_dir() {
    [[ -d "$1" ]] || kws_die "required directory missing: $1"
}

kws_run() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

kws_env_ready() {
    mkdir -p "$PYTHONPYCACHEPREFIX" "$DOWNLOADS" "$EVAL_ROOT" "$EXPORT_DIR"
    [[ -x "$PYTHON_BIN" ]] || kws_die "python not found or not executable: $PYTHON_BIN"
}
