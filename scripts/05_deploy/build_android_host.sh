#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

bash "$WORKSPACE_ROOT/android/tests/build_host.sh"
