#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for scripts/04_eval/infer_webui.py."""

import runpy
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "scripts/04_eval/infer_webui.py"

if __name__ == "__main__":
    sys.argv[0] = str(_IMPL)
    runpy.run_path(str(_IMPL), run_name="__main__")
