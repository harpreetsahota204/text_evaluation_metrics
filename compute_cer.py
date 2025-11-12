"""
Character Error Rate (CER) operator.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import Levenshtein

import fiftyone.operators as foo
from fiftyone.operators import types

from .base import BaseTextEvaluationOperator


class ComputeCER(BaseTextEvaluationOperator):
    """Compute Character Error Rate."""
    
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_cer",
            label="Compute CER",
            description="Compute Character Error Rate - ratio of character edits needed",
            icon="/assets/spell-check-svgrepo-com.svg",
            dynamic=True,
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
        Compute Character Error Rate for text fields.
        
        Args:
            sample_collection: A FiftyOne dataset or view
            pred_field (str): Name of the field containing predictions
            gt_field (str): Name of the field containing ground truth
            output_field (str, optional): Name for the output field. 
                Defaults to "{pred_field}_cer"
            case_sensitive (bool, optional): Use case-sensitive comparison. 
                Defaults to True
            delegate (bool, optional): Request delegated execution. Defaults to False
            
        Returns:
            dict: Execution results containing mean_cer and samples_evaluated
        """
        if output_field is None:
            output_field = f"{pred_field}_cer"
        
        ctx = dict(view=sample_collection.view())
        params = dict(
            pred_field=pred_field,
            gt_field=gt_field,
            output_field=output_field,
            case_sensitive=case_sensitive,
        )
        
        return foo.execute_operator(
            self.uri, 
            ctx, 
            params=params,
            request_delegation=delegate,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        if not self._build_field_inputs(inputs, ctx):
            return types.Property(inputs)
        
        pred_field = ctx.params.get("pred_field", "")
        default_output = f"{pred_field}_cer" if pred_field else "cer"
        inputs.str(
            "output_field",
            default=default_output,
        )
        
        inputs.view(
            "header",
            types.Header(label="CER Parameters", divider=True)
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
            self._compute_cer(gt, pred, case_sensitive)
            for gt, pred in zip(gt_strings, pred_strings)
        ]
        
        mean_score = self._save_scores(ctx, target_view, output_field, scores)
        
        return {
            "output_field": output_field,
            "samples_evaluated": len(target_view),
            "mean_cer": mean_score,
        }
    
    @staticmethod
    def _compute_cer(gt, pred, case_sensitive=True):
        """Compute Character Error Rate."""
        if len(gt) == 0:
            return 1.0 if len(pred) > 0 else 0.0
        
        gt_norm = gt if case_sensitive else gt.lower()
        pred_norm = pred if case_sensitive else pred.lower()
        
        distance = Levenshtein.distance(gt_norm, pred_norm)
        return distance / len(gt)

