"""
ANLS (Average Normalized Levenshtein Similarity) operator.

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
    threshold,
    case_sensitive,
    delegate,
):
    """Handle calling the operator programmatically."""
    ctx = dict(dataset=sample_collection)
    params = dict(
        pred_field=pred_field,
        gt_field=gt_field,
        output_field=output_field,
        threshold=threshold,
        case_sensitive=case_sensitive,
    )
    return foo.execute_operator(
        uri,
        ctx,
        params=params,
        delegate=delegate,
    )


class ComputeANLS(BaseTextEvaluationOperator):
    """Compute ANLS (Average Normalized Levenshtein Similarity) scores."""
    
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_anls",
            label="Compute ANLS",
            description="Compute ANLS (Average Normalized Levenshtein Similarity) - primary metric for VLM OCR evaluation",
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
        threshold=0.5,
        case_sensitive=False,
        delegate=False,
    ):
        """
        Compute ANLS scores for text fields.
        
        Args:
            sample_collection: A FiftyOne dataset or view
            pred_field (str): Name of the field containing predictions
            gt_field (str): Name of the field containing ground truth
            output_field (str, optional): Name for the output field. 
                Defaults to "{pred_field}_anls"
            threshold (float, optional): ANLS threshold (0.0-1.0). Defaults to 0.5
            case_sensitive (bool, optional): Use case-sensitive comparison. 
                Defaults to False
            delegate (bool, optional): Request delegated execution. Defaults to False
            
        Returns:
            dict: Execution results containing mean_anls and samples_evaluated
        """
        if output_field is None:
            output_field = f"{pred_field}_anls"
        
        return _handle_calling(
            self.uri,
            sample_collection,
            pred_field,
            gt_field,
            output_field,
            threshold,
            case_sensitive,
            delegate,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        if not self._build_field_inputs(inputs, ctx):
            return types.Property(inputs)
        
        # Set intelligent default for output field
        pred_field = ctx.params.get("pred_field", "")
        default_output = f"{pred_field}_anls" if pred_field else "anls"
        inputs.str(
            "output_field",
            label="Output Field Name",
            description="Name for the computed metric field",
            default=default_output,
            required=True,
        )
        
        inputs.view(
            "header",
            types.Header(label="ANLS Parameters", divider=True)
        )
        
        inputs.float(
            "threshold",
            label="ANLS Threshold",
            description="Minimum similarity to count as correct (0.0-1.0)",
            default=0.5,
            min=0.0,
            max=1.0,
            required=True,
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
        
        threshold = ctx.params.get("threshold", 0.5)
        case_sensitive = ctx.params.get("case_sensitive", False)
        output_field = ctx.params.get("output_field")
        
        scores = [
            self._compute_anls(gt, pred, threshold, case_sensitive)
            for gt, pred in zip(gt_strings, pred_strings)
        ]
        
        mean_score = self._save_scores(ctx, target_view, output_field, scores)
        
        return {
            "output_field": output_field,
            "samples_evaluated": len(target_view),
            "mean_anls": mean_score,
            "threshold": threshold,
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
    def _compute_anls(cls, gt, pred, threshold=0.5, case_sensitive=False):
        """Compute ANLS score with threshold."""
        similarity = cls._normalized_levenshtein_similarity(
            gt.strip(), 
            pred.strip(), 
            case_sensitive
        )
        return similarity if similarity >= threshold else 0.0

