#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

"$PYTHON_BIN" - <<'PY'
import sys
import torch
import torchaudio

expected = "2.5.1"
for name, mod in (("torch", torch), ("torchaudio", torchaudio)):
    version = mod.__version__
    if not str(version).startswith(expected):
        sys.exit(f"ERROR: {name} must be {expected}, got {version}")
if not hasattr(torchaudio, "info"):
    sys.exit("ERROR: torchaudio.info missing; install torchaudio 2.5.1")
print(f"torch={torch.__version__}")
print(f"torchaudio={torchaudio.__version__}")
print("torchaudio.info=ok")
PY

command -v sox >/dev/null 2>&1 || kws_die "sox not found; install via conda-forge"
sox --version | head -n 1
