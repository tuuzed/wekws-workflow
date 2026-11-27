#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for scripts/04_eval/infer_core.py."""

import runpy
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "scripts/04_eval/infer_core.py"
_IMPL_DIR = str(_IMPL.parent)

# Keep sibling imports resolvable when this compatibility wrapper is imported
# from the repository root.
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

_NS = runpy.run_path(str(_IMPL))

KwsEngine = _NS["KwsEngine"]
list_epoch_checkpoints = _NS["list_epoch_checkpoints"]
list_named_checkpoints = _NS["list_named_checkpoints"]

__all__ = ["KwsEngine", "list_epoch_checkpoints", "list_named_checkpoints"]
