"""
Word Error Rate (WER) operator.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import Levenshtein

import fiftyone.operators as foo
from fiftyone.operators import types

from .base import BaseTextEvaluationOperator


def _handle_calling(
    uri,
    sample_collection,
    pred_field,
    gt_field,
    output_field,
    case_sensitive,
    delegate,
):
    """Handle calling the operator programmatically."""
    ctx = dict(dataset=sample_collection)
    params = dict(
        pred_field=pred_field,
        gt_field=gt_field,
        output_field=output_field,
        case_sensitive=case_sensitive,
    )
    return foo.execute_operator(
        uri,
        ctx,
        params=params,
        delegate=delegate,
    )


class ComputeWER(BaseTextEvaluationOperator):
    """Compute Word Error Rate."""
    
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_wer",
            label="Compute WER",
            description="Compute Word Error Rate - ratio of word edits needed",
            icon="/assets/spell-check-svgrepo-com.svg",
            dynamic=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )
    
    def __call__(
        self,
        sample_collection,
        pred_field,
        gt_field,
        output_field=None,
        case_sensitive=True,
        delegate=False,
    ):
        """
        Compute Word Error Rate for text fields.
        
        Args:
            sample_collection: A FiftyOne dataset or view
            pred_field (str): Name of the field containing predictions
            gt_field (str): Name of the field containing ground truth
            output_field (str, optional): Name for the output field. 
                Defaults to "{pred_field}_wer"
            case_sensitive (bool, optional): Use case-sensitive comparison. 
                Defaults to True
            delegate (bool, optional): Request delegated execution. Defaults to False
            
        Returns:
            dict: Execution results containing mean_wer and samples_evaluated
        """
        if output_field is None:
            output_field = f"{pred_field}_wer"
        
        return _handle_calling(
            self.uri,
            sample_collection,
            pred_field,
            gt_field,
            output_field,
            case_sensitive,
            delegate,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        if not self._build_field_inputs(inputs, ctx):
            return types.Property(inputs)
        
        pred_field = ctx.params.get("pred_field", "")
        default_output = f"{pred_field}_wer" if pred_field else "wer"
        inputs.str(
            "output_field",
            default=default_output,
        )
        
        inputs.view(
            "header",
            types.Header(label="WER Parameters", divider=True)
        )
        
        inputs.bool(
            "case_sensitive",
            label="Case Sensitive",
            description="Use case-sensitive comparison",
            default=True,
        )
        
        return types.Property(inputs)
    
    def execute(self, ctx):
        gt_strings, pred_strings, target_view = self._get_text_pairs(ctx)
        
        case_sensitive = ctx.params.get("case_sensitive", True)
        output_field = ctx.params.get("output_field")
        
        scores = [
            self._compute_wer(gt, pred, case_sensitive)
            for gt, pred in zip(gt_strings, pred_strings)
        ]
        
        mean_score = self._save_scores(ctx, target_view, output_field, scores)
        
        return {
            "output_field": output_field,
            "samples_evaluated": len(target_view),
            "mean_wer": mean_score,
        }
    
    @staticmethod
    def _compute_wer(gt, pred, case_sensitive=True):
        """Compute Word Error Rate."""
        gt_words = gt.split()
        pred_words = pred.split()
        
        if len(gt_words) == 0:
            return 1.0 if len(pred_words) > 0 else 0.0
        
        if not case_sensitive:
            gt_words = [w.lower() for w in gt_words]
            pred_words = [w.lower() for w in pred_words]
        
        distance = Levenshtein.distance(gt_words, pred_words)
        return distance / len(gt_words)

