"""
Base operator class for text evaluation.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types


class BaseTextEvaluationOperator(foo.Operator):
    """Base class for text evaluation operators."""
    
    def _get_string_fields(self, ctx):
        """Get all StringFields from the dataset schema."""
        schema = ctx.dataset.get_field_schema()
        return [
            types.Choice(label=name, value=name)
            for name, field in schema.items()
            if isinstance(field, fo.StringField)
        ]
    
    def _build_field_inputs(self, inputs, ctx):
        """Build common field selection inputs."""
        string_fields = self._get_string_fields(ctx)
        
        if not string_fields:
            inputs.view(
                "warning",
                types.Warning(
                    label="No StringFields Found",
                    description="Dataset must have StringFields to evaluate"
                )
            )
            return False
        
        inputs.str(
            "pred_field",
            label="Prediction Field",
            description="StringField containing model predictions",
            required=True,
            view=types.AutocompleteView(choices=string_fields)
        )
        
        inputs.str(
            "gt_field",
            label="Ground Truth Field",
            description="StringField containing ground truth text",
            required=True,
            view=types.AutocompleteView(choices=string_fields)
        )
        
        inputs.str(
            "output_field",
            label="Output Field Name",
            description="Name for the computed metric field",
            required=True,
        )
        
        return True
    
    def _get_text_pairs(self, ctx):
        """Extract prediction and ground truth text pairs from the target view."""
        pred_field = ctx.params.get("pred_field")
        gt_field = ctx.params.get("gt_field")
        target_view = ctx.target_view()
        
        gt_strings = target_view.values(gt_field)
        pred_strings = target_view.values(pred_field)
        
        return gt_strings, pred_strings, target_view
    
    def _save_scores(self, ctx, target_view, output_field, scores):
        """Save computed scores to the dataset."""
        ctx.dataset.add_sample_field(output_field, fo.FloatField)
        target_view.set_values(output_field, scores)
        ctx.dataset.save()
        
        return self._safe_mean(scores)
    
    @staticmethod
    def _safe_mean(values):
        """Compute mean, filtering out None values."""
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else None
    
    def resolve_delegation(self, ctx):
        """Delegate for large datasets."""
        return len(ctx.target_view()) > 1000

