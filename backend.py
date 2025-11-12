"""
Custom evaluation backend for StringField (text) evaluation in FiftyOne.

This backend allows evaluate_regressions() to work with StringFields instead
of Regression labels, enabling text-to-text evaluation metrics.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone.utils.eval.regression as fouer


class TextEvaluationConfig(fouer.RegressionEvaluationConfig):
    """
    Configuration for text evaluation using StringFields.
    
    This config allows using StringFields (plain text) instead of
    Regression labels in evaluation workflows.
    
    Args:
        pred_field (str): name of the StringField containing predictions
        gt_field (str): name of the StringField containing ground truth
        **kwargs: additional arguments passed to parent config
    
    Example:
        ```python
        import fiftyone as fo
        
        dataset = fo.Dataset("text_eval")
        # Add samples with StringFields...
        
        results = dataset.evaluate_regressions(
            "prediction",           # StringField
            gt_field="ground_truth", # StringField
            eval_key="eval",
            method="text",          # Use this backend
            custom_metrics=["@yourusername/text-evaluation-metrics/anls"],
        )
        ```
    """
    
    def __init__(self, pred_field, gt_field="ground_truth", **kwargs):
        super().__init__(pred_field=pred_field, gt_field=gt_field, **kwargs)


class TextEvaluation(fouer.RegressionEvaluation):
    """
    Evaluation backend for StringFields (text data).
    
    This backend extracts string values from StringFields and makes them
    available to custom metrics via the standard evaluation results object.
    
    Unlike standard regression evaluation which expects Regression labels
    with numeric values, this backend works directly with StringFields
    containing text data.
    
    The extracted strings are provided to custom metrics via:
        - results.ytrue: List of ground truth strings
        - results.ypred: List of prediction strings
    
    Custom metrics can then compute text similarity, edit distance, BLEU,
    ROUGE, or any other text-based metrics.
    """
    
    def evaluate_samples(self, samples, eval_key=None, **kwargs):
        """
        Evaluates text predictions by extracting string values.
        
        This method extracts the actual string content from StringFields
        (not Regression labels) and makes them available to custom metrics
        through the results object.
        
        Args:
            samples: SampleCollection to evaluate
            eval_key (str): evaluation key for field naming
            **kwargs: additional evaluation parameters
        
        Returns:
            RegressionResults object containing:
                - ytrue: list of ground truth strings
                - ypred: list of prediction strings
                - samples: the sample collection
                - config: evaluation configuration
        """
        pred_field = self.config.pred_field
        gt_field = self.config.gt_field
        
        # Extract the ACTUAL STRING VALUES from StringFields
        # These are strings, not numeric regression values
        ytrue = samples.values(gt_field)
        ypred = samples.values(pred_field)
        
        # Create results object that custom metrics can access
        # Custom metrics will receive these strings via:
        #   - results.ytrue
        #   - results.ypred
        results = fouer.RegressionResults(
            samples,
            self.config,
            eval_key,
            ytrue=ytrue,    # List of strings (ground truth)
            ypred=ypred,    # List of strings (predictions)
            missing=None,   # No missing value handling for text
            backend=self,
        )
        
        return results

