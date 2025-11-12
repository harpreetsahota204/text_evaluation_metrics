"""
Text evaluation for FiftyOne with proper evaluation framework integration.

This module provides the evaluate_text() function that works with StringFields
while properly integrating with FiftyOne's evaluation framework.
"""
import fiftyone as fo
from .text_evaluation_backend import TextEvaluation, TextEvaluationConfig


def evaluate_text(
    samples,
    pred_field,
    gt_field="ground_truth",
    eval_key="eval",
    custom_metrics=None,
    **kwargs
):
    """
    Evaluate text predictions against ground truth using StringFields.
    
    This function provides proper FiftyOne evaluation integration:
    - Works with StringFields (no Label conversion needed)
    - Integrates with FiftyOne's evaluation system
    - Supports custom metric operators
    - Enables delete_evaluation() cleanup
    - Works with evaluation UI
    
    Args:
        samples: FiftyOne Dataset or DatasetView to evaluate
        pred_field: name of prediction field (StringField)
        gt_field: name of ground truth field (StringField)
        eval_key: evaluation key for this evaluation run
        custom_metrics: list of custom metric operator URIs or dict mapping
            URIs to config dicts. Examples:
            - ["@namespace/plugin/metric1", "@namespace/plugin/metric2"]
            - {"@namespace/plugin/metric1": {"threshold": 0.5}}
        **kwargs: additional arguments passed to custom metrics
    
    Returns:
        TextResults object with evaluation results
    
    Example:
        >>> results = evaluate_text(
        ...     dataset,
        ...     "prediction",
        ...     gt_field="ground_truth",
        ...     eval_key="eval",
        ...     custom_metrics=[
        ...         "@harpreetsahota/text-evaluation-metrics/anls",
        ...         "@harpreetsahota/text-evaluation-metrics/exact_match",
        ...     ],
        ... )
        >>> results.print_report()
        >>> dataset.delete_evaluation("eval")
    """
    # Create evaluation config
    config = TextEvaluationConfig(
        pred_field=pred_field,
        gt_field=gt_field,
        eval_key=eval_key,
        custom_metrics=custom_metrics,
    )
    
    # Create evaluation instance
    evaluation = TextEvaluation(config)
    
    # Run evaluation
    results = evaluation.evaluate_samples(samples, eval_key=eval_key, **kwargs)
    
    # Run custom metrics if provided
    if custom_metrics:
        results = _run_custom_metrics(
            samples,
            results,
            custom_metrics,
            **kwargs
        )
    
    # Save evaluation info to dataset for cleanup/rename support
    dataset = samples._dataset if hasattr(samples, '_dataset') else samples
    dataset._register_evaluation(eval_key, results)
    
    return results


def _run_custom_metrics(samples, results, custom_metrics, **kwargs):
    """Run custom metric operators on the evaluation results."""
    import fiftyone.operators as foo
    
    # Normalize custom_metrics to dict format
    if isinstance(custom_metrics, (list, tuple)):
        custom_metrics = {metric: {} for metric in custom_metrics}
    
    # Run each custom metric
    for metric_uri, metric_kwargs in custom_metrics.items():
        # Merge kwargs
        merged_kwargs = {**kwargs, **metric_kwargs}
        
        # Get the operator
        operator = foo.get_operator(metric_uri)
        
        # Run the metric
        metric_value = operator.compute(samples, results, **merged_kwargs)
        
        # Store aggregate in results
        if metric_value is not None and hasattr(results, 'add_metric'):
            # Get the aggregate key from operator config
            aggregate_key = operator.config.aggregate_key
            results.add_metric(aggregate_key, metric_value)
    
    return results

