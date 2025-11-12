"""
Text evaluation plugin for FiftyOne.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from .operators import (
    ComputeANLS,
    ComputeExactMatch,
    ComputeNormalizedSimilarity,
    ComputeCER,
    ComputeWER,
)


def register(p):
    """Register all text evaluation operators."""
    p.register(ComputeANLS)
    p.register(ComputeExactMatch)
    p.register(ComputeNormalizedSimilarity)
    p.register(ComputeCER)
    p.register(ComputeWER)
