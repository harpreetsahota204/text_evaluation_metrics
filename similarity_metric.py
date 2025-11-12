"""
Normalized Similarity metric for FiftyOne.

Continuous similarity measure without threshold, useful for
analyzing error patterns and model behavior.
"""
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .metrics import normalized_levenshtein_similarity


class NormalizedSimilarityMetric(foo.EvaluationMetric):
    """
    Computes normalized Levenshtein similarity without threshold.
    
    Unlike ANLS which applies a threshold, this returns the raw similarity
    score between 0.0 and 1.0. Useful for continuous analysis of model
    performance without the binary threshold cutoff.
    """
    
    @property
    def config(self):
        return foo.EvaluationMetricConfig(
            name="normalized_similarity",
            label="Normalized Similarity",
            description="Normalized Levenshtein similarity without threshold (continuous 0-1)",
            aggregate_key="mean_similarity",
            lower_is_better=False,
        )
    
    def compute(self, samples, results):
        """
        Compute normalized similarity for each sample and return mean.
        
        Args:
            samples: Sample collection to evaluate
            results: Evaluation results object containing configuration
        
        Returns:
            Mean similarity score across all samples
        """
        # Get evaluation configuration
        eval_key = results.key
        gt_field = results.config.gt_field
        pred_field = results.config.pred_field
        
        # Extract text strings from samples
        gt_strings = samples.values(gt_field)
        pred_strings = samples.values(pred_field)
        
        # Compute similarity for each sample
        scores = []
        for gt, pred in zip(gt_strings, pred_strings):
            if gt is None or pred is None:
                scores.append(None)
            else:
                score = normalized_levenshtein_similarity(
                    gt.strip(), 
                    pred.strip()
                )
                scores.append(score)
        
        # Store per-sample similarity scores
        metric_field = f"{eval_key}_normalized_similarity"
        dataset = samples._dataset
        dataset.add_sample_field(metric_field, fo.FloatField)
        samples.set_values(metric_field, scores)
        
        # Return mean similarity
        valid_scores = [s for s in scores if s is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else None
    
    def get_fields(self, samples, config, eval_key):
        """Return list of fields created by this metric."""
        return [f"{eval_key}_normalized_similarity"]

