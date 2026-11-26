#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

export CHECKPOINT="${CHECKPOINT:-$DEPLOY_CHECKPOINT}"
bash "$WORKSPACE_ROOT/android/tests/run_compare.sh" "$@"
