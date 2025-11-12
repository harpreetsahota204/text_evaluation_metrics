"""
ANLS metric operator.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .metrics import compute_anls, safe_mean


class ANLSMetric(foo.EvaluationMetric):
    @property
    def config(self):
        return foo.EvaluationMetricConfig(
            name="anls",
            label="ANLS Score",
            description="Average Normalized Levenshtein Similarity (standard VLM OCR metric)",
            aggregate_key="mean_anls",
            lower_is_better=False,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.float(
            "threshold",
            label="ANLS Threshold",
            description="Minimum similarity to count as correct",
            default=0.5,
            required=True,
        )
        inputs.bool(
            "case_sensitive",
            label="Case Sensitive",
            default=False,
            required=False,
        )
        return types.Property(inputs)

    def compute(self, samples, results, threshold=0.5, case_sensitive=False):
        eval_key = results.key
        ytrue = results.ytrue
        ypred = results.ypred
        
        scores = [
            compute_anls(yt, yp, threshold, case_sensitive)
            for yt, yp in zip(ytrue, ypred)
        ]
        
        metric_field = f"{eval_key}_anls"
        samples._dataset.add_sample_field(metric_field, fo.FloatField)
        samples.set_values(metric_field, scores)
        
        return safe_mean(scores)

    def get_fields(self, samples, config, eval_key):
        return [f"{eval_key}_anls"]

