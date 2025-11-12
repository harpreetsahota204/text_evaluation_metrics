"""
Text evaluation metrics for FiftyOne.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from .anls import ANLSMetric
from .exact_match import ExactMatchMetric
from .normalized_similarity import NormalizedSimilarityMetric
from .cer import CERMetric
from .wer import WERMetric


def register(p):
    p.register(ANLSMetric)
    p.register(ExactMatchMetric)
    p.register(NormalizedSimilarityMetric)
    p.register(CERMetric)
    p.register(WERMetric)
