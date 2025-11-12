"""
Text evaluation operator.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .metrics import (
    compute_anls,
    compute_exact_match,
    compute_normalized_similarity,
    compute_cer,
    compute_wer,
    safe_mean,
)


class EvaluateText(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="evaluate_text",
            label="Evaluate Text Fields",
            description="Compute text evaluation metrics (ANLS, Exact Match, CER, WER)",
            icon="/assets/spell-check-svgrepo-com.svg",
            dynamic=True,
        )

    def __call__(
        self,
        sample_collection,
        pred_field,
        gt_field,
        eval_key="eval",
        compute_anls=True,
        compute_exact_match=True,
        compute_similarity=False,
        compute_cer=False,
        compute_wer=False,
        anls_threshold=0.5,
        case_sensitive=False,
        delegate=False,
    ):
        ctx = dict(view=sample_collection.view())
        params = dict(
            pred_field=pred_field,
            gt_field=gt_field,
            eval_key=eval_key,
            compute_anls=compute_anls,
            compute_exact_match=compute_exact_match,
            compute_similarity=compute_similarity,
            compute_cer=compute_cer,
            compute_wer=compute_wer,
            anls_threshold=anls_threshold,
            case_sensitive=case_sensitive,
        )
        return foo.execute_operator(
            self.uri,
            ctx,
            params=params,
            request_delegation=delegate
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        schema = ctx.dataset.get_field_schema()
        string_fields = [
            types.Choice(label=name, value=name)
            for name, field in schema.items()
            if isinstance(field, fo.StringField)
        ]

        if not string_fields:
            inputs.view(
                "warning",
                types.Warning(
                    label="No StringFields Found",
                    description="Dataset must have StringFields to evaluate"
                )
            )
            return types.Property(inputs)

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
            "eval_key",
            label="Evaluation Key",
            description="Prefix for metric field names",
            default="eval",
            required=True
        )

        inputs.view(
            "header",
            types.Header(
                label="Metrics to Compute",
                divider=True
            )
        )

        inputs.bool(
            "compute_anls",
            label="ANLS (Primary metric for VLM OCR)",
            default=True
        )

        inputs.bool(
            "compute_exact_match",
            label="Exact Match Accuracy",
            default=True
        )

        inputs.bool(
            "compute_similarity",
            label="Normalized Similarity (no threshold)",
            default=False
        )

        inputs.bool(
            "compute_cer",
            label="Character Error Rate",
            default=False
        )

        inputs.bool(
            "compute_wer",
            label="Word Error Rate",
            default=False
        )

        inputs.view(
            "header2",
            types.Header(
                label="Parameters",
                divider=True
            )
        )

        inputs.float(
            "anls_threshold",
            label="ANLS Threshold",
            description="Minimum similarity to count as correct",
            default=0.5,
            min=0.0,
            max=1.0
        )

        inputs.bool(
            "case_sensitive",
            label="Case Sensitive",
            description="Use case-sensitive comparison",
            default=False
        )

        return types.Property(inputs)

    def execute(self, ctx):
        pred_field = ctx.params.get("pred_field")
        gt_field = ctx.params.get("gt_field")
        eval_key = ctx.params.get("eval_key", "eval")

        target_view = ctx.target_view()
        
        gt_strings = target_view.values(gt_field)
        pred_strings = target_view.values(pred_field)

        anls_threshold = ctx.params.get("anls_threshold", 0.5)
        case_sensitive = ctx.params.get("case_sensitive", False)

        results = {}

        if ctx.params.get("compute_anls", True):
            scores = [
                compute_anls(gt, pred, anls_threshold, case_sensitive)
                for gt, pred in zip(gt_strings, pred_strings)
            ]
            field_name = f"{eval_key}_anls"
            ctx.dataset.add_sample_field(field_name, fo.FloatField)
            target_view.set_values(field_name, scores)
            results["mean_anls"] = safe_mean(scores)

        if ctx.params.get("compute_exact_match", True):
            scores = [
                compute_exact_match(gt, pred, case_sensitive, True)
                for gt, pred in zip(gt_strings, pred_strings)
            ]
            field_name = f"{eval_key}_exact_match"
            ctx.dataset.add_sample_field(field_name, fo.FloatField)
            target_view.set_values(field_name, scores)
            results["accuracy"] = safe_mean(scores)

        if ctx.params.get("compute_similarity", False):
            scores = [
                compute_normalized_similarity(gt, pred, case_sensitive)
                for gt, pred in zip(gt_strings, pred_strings)
            ]
            field_name = f"{eval_key}_normalized_similarity"
            ctx.dataset.add_sample_field(field_name, fo.FloatField)
            target_view.set_values(field_name, scores)
            results["mean_similarity"] = safe_mean(scores)

        if ctx.params.get("compute_cer", False):
            scores = [
                compute_cer(gt, pred, case_sensitive)
                for gt, pred in zip(gt_strings, pred_strings)
            ]
            field_name = f"{eval_key}_cer"
            ctx.dataset.add_sample_field(field_name, fo.FloatField)
            target_view.set_values(field_name, scores)
            results["mean_cer"] = safe_mean(scores)

        if ctx.params.get("compute_wer", False):
            scores = [
                compute_wer(gt, pred, case_sensitive)
                for gt, pred in zip(gt_strings, pred_strings)
            ]
            field_name = f"{eval_key}_wer"
            ctx.dataset.add_sample_field(field_name, fo.FloatField)
            target_view.set_values(field_name, scores)
            results["mean_wer"] = safe_mean(scores)

        ctx.dataset.save()

        return {
            "eval_key": eval_key,
            "samples_evaluated": len(target_view),
            "metrics": results
        }

