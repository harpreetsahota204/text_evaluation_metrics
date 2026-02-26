"""
Normalized Similarity operator.

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


class ComputeNormalizedSimilarity(BaseTextEvaluationOperator):
    """Compute normalized Levenshtein similarity without threshold."""
    
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_normalized_similarity",
            label="Compute Normalized Similarity",
            description="Compute continuous normalized Levenshtein similarity (0.0-1.0) without threshold",
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
        delegate=False,
    ):
        """
        Compute normalized similarity for text fields.
        
        Args:
            sample_collection: A FiftyOne dataset or view
            pred_field (str): Name of the field containing predictions
            gt_field (str): Name of the field containing ground truth
            output_field (str, optional): Name for the output field. 
                Defaults to "{pred_field}_similarity"
            case_sensitive (bool, optional): Use case-sensitive comparison. 
                Defaults to False
            delegate (bool, optional): Request delegated execution. Defaults to False
            
        Returns:
            dict: Execution results containing mean_similarity and samples_evaluated
        """
        if output_field is None:
            output_field = f"{pred_field}_similarity"
        
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
        default_output = f"{pred_field}_similarity" if pred_field else "similarity"
        inputs.str(
            "output_field",
            label="Output Field Name",
            description="Name for the computed metric field",
            default=default_output,
            required=True,
        )
        
        inputs.view(
            "header",
            types.Header(label="Similarity Parameters", divider=True)
        )
        
        inputs.bool(
            "case_sensitive",
            label="Case Sensitive",
            description="Use case-sensitive comparison",
            default=False,
        )
        
        return types.Property(inputs)
    
    def execute(self, ctx):
        gt_strings, pred_strings, target_view = self._get_text_pairs(ctx)
        
        case_sensitive = ctx.params.get("case_sensitive", False)
        output_field = ctx.params.get("output_field")
        
        scores = [
            self._compute_normalized_similarity(gt, pred, case_sensitive)
            for gt, pred in zip(gt_strings, pred_strings)
        ]
        
        mean_score = self._save_scores(ctx, target_view, output_field, scores)
        
        return {
            "output_field": output_field,
            "samples_evaluated": len(target_view),
            "mean_similarity": mean_score,
        }
    
    @staticmethod
    def _normalized_levenshtein_similarity(s1, s2, case_sensitive=False):
        """Calculate normalized Levenshtein similarity (0.0 to 1.0)."""
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        
        str1 = s1 if case_sensitive else s1.lower()
        str2 = s2 if case_sensitive else s2.lower()
        
        distance = Levenshtein.distance(str1, str2)
        max_len = max(len(str1), len(str2))
        similarity = 1.0 - (distance / max_len)
        
        return max(0.0, similarity)
    
    @classmethod
    def _compute_normalized_similarity(cls, gt, pred, case_sensitive=False):
        """Compute normalized similarity without threshold."""
        return cls._normalized_levenshtein_similarity(gt.strip(), pred.strip(), case_sensitive)

