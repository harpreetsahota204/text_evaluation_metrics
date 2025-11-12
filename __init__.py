"""
Text evaluation plugin for FiftyOne.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from .evaluate_text import EvaluateText


def register(p):
    p.register(EvaluateText)
