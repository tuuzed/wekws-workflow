#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for scripts/04_eval/infer_webui.py."""

import runpy
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "scripts/04_eval/infer_webui.py"
_IMPL_DIR = str(_IMPL.parent)

# runpy.run_path() does not put the implementation directory on sys.path, but
# infer_webui.py imports sibling modules (infer_core, onnx_kws_core) by name.
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

if __name__ == "__main__":
    sys.argv[0] = str(_IMPL)
    runpy.run_path(str(_IMPL), run_name="__main__")
