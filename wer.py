"""
WER (Word Error Rate) metric operator.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .metrics import compute_wer, safe_mean


class WERMetric(foo.EvaluationMetric):
    @property
    def config(self):
        return foo.EvaluationMetricConfig(
            name="wer",
            label="Word Error Rate",
            description="Word-level edit distance",
            aggregate_key="mean_wer",
            lower_is_better=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.bool(
            "case_sensitive",
            label="Case Sensitive",
            default=True,
            required=False,
        )
        return types.Property(inputs)

    def compute(self, samples, results, case_sensitive=True):
        eval_key = results.key
        ytrue = results.ytrue
        ypred = results.ypred
        
        scores = [
            compute_wer(yt, yp, case_sensitive)
            for yt, yp in zip(ytrue, ypred)
        ]
        
        metric_field = f"{eval_key}_wer"
        samples._dataset.add_sample_field(metric_field, fo.FloatField)
        samples.set_values(metric_field, scores)
        
        return safe_mean(scores)

    def get_fields(self, samples, config, eval_key):
        return [f"{eval_key}_wer"]

