#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

ANDROID_DIR="$WORKSPACE_ROOT/android"
SDK_ROOT="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"

if [[ -z "$SDK_ROOT" || ! -d "$SDK_ROOT" ]]; then
    echo "ERROR: Android SDK not found. Set ANDROID_HOME or ANDROID_SDK_ROOT." >&2
    exit 1
fi

ANDROID_JAR="$(find "$SDK_ROOT/platforms" -maxdepth 2 -name android.jar -type f 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -z "$ANDROID_JAR" ]]; then
    echo "ERROR: android.jar not found under $SDK_ROOT/platforms." >&2
    exit 1
fi

echo "Using Android SDK: $SDK_ROOT"
echo "Using android.jar:  $ANDROID_JAR"
cd "$ANDROID_DIR"
./gradlew assembleDebug "$@"
