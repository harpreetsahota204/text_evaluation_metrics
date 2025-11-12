"""
ANLS (Average Normalized Levenshtein Similarity) metric for FiftyOne.

The gold standard metric for VLM OCR evaluation, used in DocVQA,
OCRBench, TextVQA, and other modern benchmarks.
"""
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .metrics import anls as compute_anls


class ANLSMetric(foo.EvaluationMetric):
    """
    Computes ANLS score for text evaluation.
    
    ANLS measures similarity between ground truth and prediction using
    normalized Levenshtein distance, with a threshold to distinguish
    acceptable from unacceptable predictions.
    """
    
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
        """Define configurable parameters for the metric."""
        inputs = types.Object()
        
        inputs.float(
            "threshold",
            label="ANLS Threshold",
            description="Minimum similarity to count as correct (standard: 0.5)",
            default=0.5,
            required=True,
        )
        
        return types.Property(inputs)
    
    def compute(self, samples, results, threshold=0.5):
        """
        Compute ANLS score for each sample and return aggregate.
        
        Args:
            samples: Sample collection to evaluate
            results: Evaluation results object containing configuration
            threshold: Minimum similarity to count as correct
        
        Returns:
            Mean ANLS score across all samples
        """
        # Get evaluation configuration
        eval_key = results.key
        gt_field = results.config.gt_field
        pred_field = results.config.pred_field
        
        # Extract text strings from samples
        gt_strings = samples.values(gt_field)
        pred_strings = samples.values(pred_field)
        
        # Compute ANLS for each sample
        scores = []
        for gt, pred in zip(gt_strings, pred_strings):
            if gt is None or pred is None:
                scores.append(None)
            else:
                score = compute_anls(gt, pred, threshold)
                scores.append(score)
        
        # Store per-sample scores
        metric_field = f"{eval_key}_anls"
        dataset = samples._dataset
        dataset.add_sample_field(metric_field, fo.FloatField)
        samples.set_values(metric_field, scores)
        
        # Return aggregate (mean of non-None values)
        valid_scores = [s for s in scores if s is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else None
    
    def get_fields(self, samples, config, eval_key):
        """Return list of fields created by this metric."""
        return [f"{eval_key}_anls"]

