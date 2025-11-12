"""
Word Error Rate (WER) metric for FiftyOne.

Traditional OCR metric measuring word-level accuracy.
Useful for comparison with classical OCR systems and
analyzing errors at word granularity.
"""
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .metrics import word_error_rate


class WERMetric(foo.EvaluationMetric):
    """
    Computes Word Error Rate (WER) for text evaluation.
    
    WER measures the word-level edit distance normalized by
    ground truth word count. Lower values are better (0.0 = perfect).
    Traditional metric for OCR evaluation.
    """
    
    @property
    def config(self):
        return foo.EvaluationMetricConfig(
            name="wer",
            label="Word Error Rate",
            description="Word-level edit distance (traditional OCR metric)",
            aggregate_key="mean_wer",
            lower_is_better=True,  # Lower WER is better
        )
    
    def resolve_input(self, ctx):
        """Define configurable parameters for the metric."""
        inputs = types.Object()
        
        inputs.bool(
            "case_sensitive",
            label="Case Sensitive",
            description="Whether to use case-sensitive comparison",
            default=True,
        )
        
        return types.Property(inputs)
    
    def compute(self, samples, results, case_sensitive=True):
        """
        Compute WER for each sample and return mean.
        
        Args:
            samples: Sample collection to evaluate
            results: Evaluation results object containing configuration
            case_sensitive: Whether to use case-sensitive comparison
        
        Returns:
            Mean WER across all samples
        """
        # Get evaluation configuration
        eval_key = results.key
        gt_field = results.config.gt_field
        pred_field = results.config.pred_field
        
        # Extract text strings from samples
        gt_strings = samples.values(gt_field)
        pred_strings = samples.values(pred_field)
        
        # Compute WER for each sample
        scores = []
        for gt, pred in zip(gt_strings, pred_strings):
            if gt is None or pred is None:
                scores.append(None)
            else:
                # Apply case transformation if needed
                gt_text = gt if case_sensitive else gt.lower()
                pred_text = pred if case_sensitive else pred.lower()
                score = word_error_rate(gt_text, pred_text)
                scores.append(score)
        
        # Store per-sample WER values
        metric_field = f"{eval_key}_wer"
        dataset = samples._dataset
        dataset.add_sample_field(metric_field, fo.FloatField)
        samples.set_values(metric_field, scores)
        
        # Return mean WER
        valid_scores = [s for s in scores if s is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else None
    
    def get_fields(self, samples, config, eval_key):
        """Return list of fields created by this metric."""
        return [f"{eval_key}_wer"]

