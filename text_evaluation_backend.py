"""
Custom evaluation backend for text (StringField) evaluation.

This implements a proper FiftyOne evaluation backend that works with
StringFields, allowing custom metrics to integrate with FiftyOne's
evaluation framework and UI.
"""
import fiftyone.core.evaluation as foe
import fiftyone.core.validation as fov


class TextEvaluationConfig(foe.EvaluationMethodConfig):
    """Configuration for text evaluation.
    
    Args:
        pred_field: name of predicted text field (StringField)
        gt_field: name of ground truth text field (StringField) 
        eval_key: evaluation key to use for this evaluation
        custom_metrics: list of custom metric operator URIs
    """
    
    def __init__(self, pred_field, gt_field, eval_key, custom_metrics=None, **kwargs):
        super().__init__(**kwargs)
        self.pred_field = pred_field
        self.gt_field = gt_field
        self.eval_key = eval_key
        self.custom_metrics = custom_metrics or []


class TextEvaluation(foe.EvaluationMethod):
    """Evaluation method for text (StringField) comparisons.
    
    This provides a proper evaluation backend that integrates with
    FiftyOne's evaluation framework while working with StringFields.
    """
    
    def __init__(self, config):
        super().__init__(config)
    
    def ensure_requirements(self):
        """Validate that the required fields exist and are StringFields."""
        pred_field = self.config.pred_field
        gt_field = self.config.gt_field
        
        # Get the dataset
        if hasattr(self, 'samples'):
            dataset = self.samples._dataset
        else:
            # Will be set during evaluate_samples
            return
        
        # Validate fields exist
        if not dataset.has_field(pred_field):
            raise ValueError(f"Prediction field '{pred_field}' does not exist")
        if not dataset.has_field(gt_field):
            raise ValueError(f"Ground truth field '{gt_field}' does not exist")
    
    def evaluate_samples(self, samples, eval_key=None, **kwargs):
        """
        Evaluate text predictions against ground truth.
        
        Args:
            samples: sample collection to evaluate
            eval_key: evaluation key (uses config.eval_key if not provided)
            **kwargs: additional arguments passed to custom metrics
        
        Returns:
            TextResults instance
        """
        self.samples = samples
        self.ensure_requirements()
        
        eval_key = eval_key or self.config.eval_key
        
        # Extract text values  
        pred_texts = samples.values(self.config.pred_field)
        gt_texts = samples.values(self.config.gt_field)
        
        # Store for custom metrics to access
        self.pred_texts = pred_texts
        self.gt_texts = gt_texts
        
        # Create results object
        results = TextResults(
            samples,
            self.config,
            eval_key,
            pred_texts,
            gt_texts,
        )
        
        return results
    
    def get_fields(self, samples, eval_key):
        """Get list of fields that will be populated by this evaluation."""
        fields = []
        
        # Custom metrics will add their own fields
        # The base evaluation doesn't add any fields itself
        
        return fields
    
    def cleanup(self, samples, eval_key):
        """Clean up fields created by this evaluation."""
        # Custom metrics handle their own cleanup
        pass
    
    def rename(self, samples, eval_key, new_eval_key):
        """Rename fields when evaluation key changes."""
        # Custom metrics handle their own renaming
        pass


class TextResults(foe.EvaluationResults):
    """Results of text evaluation.
    
    This stores the evaluation results and provides access to
    aggregate metrics computed by custom metrics.
    """
    
    def __init__(self, samples, config, eval_key, pred_texts, gt_texts, **kwargs):
        super().__init__(samples, config, eval_key, **kwargs)
        self.pred_texts = pred_texts
        self.gt_texts = gt_texts
        self._aggregate_metrics = {}
    
    def report(self):
        """Generate evaluation report."""
        return "Text Evaluation Results\n" + "=" * 50 + "\n" + str(self.metrics())
    
    def print_report(self, **kwargs):
        """Print evaluation report."""
        print(self.report())
    
    def metrics(self):
        """Return aggregate metrics computed by custom metrics."""
        return self._aggregate_metrics
    
    def add_metric(self, key, value):
        """Add an aggregate metric (called by custom metrics)."""
        self._aggregate_metrics[key] = value

