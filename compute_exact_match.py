"""
Exact Match accuracy operator.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

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
    strip_whitespace,
    delegate,
):
    """Handle calling the operator programmatically."""
    ctx = dict(dataset=sample_collection)
    params = dict(
        pred_field=pred_field,
        gt_field=gt_field,
        output_field=output_field,
        case_sensitive=case_sensitive,
        strip_whitespace=strip_whitespace,
    )
    return foo.execute_operator(
        uri,
        ctx,
        params=params,
        delegate=delegate,
    )


class ComputeExactMatch(BaseTextEvaluationOperator):
    """Compute Exact Match accuracy scores."""
    
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_exact_match",
            label="Compute Exact Match",
            description="Compute binary exact match accuracy between prediction and ground truth",
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
        case_sensitive=False,
        strip_whitespace=True,
        delegate=False,
    ):
        """
        Compute exact match accuracy for text fields.
        
        Args:
            sample_collection: A FiftyOne dataset or view
            pred_field (str): Name of the field containing predictions
            gt_field (str): Name of the field containing ground truth
            output_field (str, optional): Name for the output field. 
                Defaults to "{pred_field}_exact_match"
            case_sensitive (bool, optional): Use case-sensitive comparison. 
                Defaults to False
            strip_whitespace (bool, optional): Strip leading/trailing whitespace. 
                Defaults to True
            delegate (bool, optional): Request delegated execution. Defaults to False
            
        Returns:
            dict: Execution results containing accuracy and samples_evaluated
        """
        if output_field is None:
            output_field = f"{pred_field}_exact_match"
        
        return _handle_calling(
            self.uri,
            sample_collection,
            pred_field,
            gt_field,
            output_field,
            case_sensitive,
            strip_whitespace,
            delegate,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        if not self._build_field_inputs(inputs, ctx):
            return types.Property(inputs)
        
        pred_field = ctx.params.get("pred_field", "")
        default_output = f"{pred_field}_exact_match" if pred_field else "exact_match"
        inputs.str(
            "output_field",
            default=default_output,
        )
        
        inputs.view(
            "header",
            types.Header(label="Exact Match Parameters", divider=True)
        )
        
        inputs.bool(
            "case_sensitive",
            label="Case Sensitive",
            description="Use case-sensitive comparison",
            default=False,
        )
        
        inputs.bool(
            "strip_whitespace",
            label="Strip Whitespace",
            description="Strip leading and trailing whitespace before comparison",
            default=True,
        )
        
        return types.Property(inputs)
    
    def execute(self, ctx):
        gt_strings, pred_strings, target_view = self._get_text_pairs(ctx)
        
        case_sensitive = ctx.params.get("case_sensitive", False)
        strip_whitespace = ctx.params.get("strip_whitespace", True)
        output_field = ctx.params.get("output_field")
        
        scores = [
            self._compute_exact_match(gt, pred, case_sensitive, strip_whitespace)
            for gt, pred in zip(gt_strings, pred_strings)
        ]
        
        mean_score = self._save_scores(ctx, target_view, output_field, scores)
        
        return {
            "output_field": output_field,
            "samples_evaluated": len(target_view),
            "accuracy": mean_score,
        }
    
    @staticmethod
    def _compute_exact_match(gt, pred, case_sensitive=False, strip_whitespace=True):
        """Compute exact match score (1.0 or 0.0)."""
        gt_norm = gt.strip() if strip_whitespace else gt
        pred_norm = pred.strip() if strip_whitespace else pred
        
        if not case_sensitive:
            gt_norm = gt_norm.lower()
            pred_norm = pred_norm.lower()
        
        return 1.0 if gt_norm == pred_norm else 0.0

