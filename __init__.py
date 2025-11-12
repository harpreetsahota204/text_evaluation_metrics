"""
Text Evaluation Metrics for FiftyOne.

A plugin providing standard VLM OCR evaluation metrics:
- ANLS (Average Normalized Levenshtein Similarity)
- Exact Match Accuracy  
- Normalized Similarity
- Character Error Rate (CER)
- Word Error Rate (WER)

Copyright 2025
"""

from .anls_metric import ANLSMetric
from .exact_match_metric import ExactMatchMetric
from .similarity_metric import NormalizedSimilarityMetric
from .cer_metric import CERMetric
from .wer_metric import WERMetric


def register(p):
    """Register all text evaluation metrics with FiftyOne."""
    p.register(ANLSMetric)
    p.register(ExactMatchMetric)
    p.register(NormalizedSimilarityMetric)
    p.register(CERMetric)
    p.register(WERMetric)

