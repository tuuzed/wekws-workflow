#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper.

The export implementation now lives at scripts/03_export/export_android_onnx.py.
This wrapper only forwards arguments so older commands keep working.
"""

import runpy
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/03_export/export_android_onnx.py"
)

if __name__ == "__main__":
    sys.argv[0] = str(SCRIPT)
    runpy.run_path(str(SCRIPT), run_name="__main__")
