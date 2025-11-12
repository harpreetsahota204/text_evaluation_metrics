"""
Exact Match Accuracy metric for FiftyOne.

Simple binary metric showing what percentage of predictions
are perfectly correct. Standard complement to ANLS in VLM papers.
"""
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .metrics import exact_match as compute_exact_match


class ExactMatchMetric(foo.EvaluationMetric):
    """
    Computes exact match accuracy for text evaluation.
    
    Returns 1.0 for perfect matches, 0.0 otherwise. Typically reported
    alongside ANLS as a stricter measure of correctness.
    """
    
    @property
    def config(self):
        return foo.EvaluationMetricConfig(
            name="exact_match",
            label="Exact Match Accuracy",
            description="Percentage of exact matches (case-insensitive, whitespace-stripped)",
            aggregate_key="accuracy",
            lower_is_better=False,
        )
    
    def resolve_input(self, ctx):
        """Define configurable parameters for the metric."""
        inputs = types.Object()
        
        inputs.bool(
            "case_sensitive",
            label="Case Sensitive",
            description="Whether to use case-sensitive comparison",
            default=False,
        )
        
        inputs.bool(
            "strip_whitespace",
            label="Strip Whitespace",
            description="Whether to strip leading/trailing whitespace",
            default=True,
        )
        
        return types.Property(inputs)
    
    def compute(self, samples, results, case_sensitive=False, strip_whitespace=True):
        """
        Compute exact match for each sample and return accuracy.
        
        Args:
            samples: Sample collection to evaluate
            results: Evaluation results object containing configuration
            case_sensitive: Whether to use case-sensitive comparison
            strip_whitespace: Whether to strip whitespace before comparison
        
        Returns:
            Accuracy (proportion of exact matches)
        """
        # Get evaluation configuration
        eval_key = results.key
        
        # Extract text strings from TextResults or samples
        if hasattr(results, 'gt_texts') and hasattr(results, 'pred_texts'):
            gt_strings = results.gt_texts
            pred_strings = results.pred_texts
        else:
            gt_field = results.config.gt_field
            pred_field = results.config.pred_field
            gt_strings = samples.values(gt_field)
            pred_strings = samples.values(pred_field)
        
        # Compute exact match for each sample
        scores = []
        for gt, pred in zip(gt_strings, pred_strings):
            if gt is None or pred is None:
                scores.append(None)
            else:
                score = compute_exact_match(gt, pred, case_sensitive, strip_whitespace)
                scores.append(score)
        
        # Store per-sample binary scores
        metric_field = f"{eval_key}_exact_match"
        dataset = samples._dataset
        dataset.add_sample_field(metric_field, fo.FloatField)
        samples.set_values(metric_field, scores)
        
        # Return accuracy (mean of binary scores)
        valid_scores = [s for s in scores if s is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else None
    
    def get_fields(self, samples, config, eval_key):
        """Return list of fields created by this metric."""
        return [f"{eval_key}_exact_match"]

