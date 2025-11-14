# Text Evaluation Metrics for FiftyOne

![image](text_eval_gif.gif)

Operator plugin for evaluating text fields (StringFields) in FiftyOne datasets with standard VLM OCR metrics.

## Installation

```bash
pip install python-Levenshtein
```

Install the plugin:

```bash
fiftyone plugins download https://github.com/harpreetsahota204/text_evaluation_metrics
```

## Overview

This plugin provides five text evaluation metrics for comparing predictions against ground truth:

| Metric | Description | Use Case | Range |
|--------|-------------|----------|-------|
| **ANLS** | Average Normalized Levenshtein Similarity with threshold | Primary OCR metric for VLMs, robust to minor errors | 0.0-1.0 |
| **Exact Match** | Binary perfect match | Strict evaluation (form fields, IDs) | 0.0 or 1.0 |
| **Normalized Similarity** | Continuous Levenshtein similarity without threshold | Fine-grained analysis and ranking | 0.0-1.0 |
| **CER** | Character Error Rate | Character-level error analysis | 0.0+ (lower is better) |
| **WER** | Word Error Rate | Word-level error analysis | 0.0+ (lower is better) |

## Available Operators

### 1. ComputeANLS (`compute_anls`)
**Average Normalized Levenshtein Similarity** - Standard metric for OCR evaluation, normalizes edit distance by string length and applies a configurable threshold (default: 0.5)

```python
operator = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_anls")
result = operator(
    dataset,
    pred_field="prediction",
    gt_field="ground_truth",
    output_field="anls_score",  # optional, defaults to "{pred_field}_anls"
    threshold=0.5,  # ANLS threshold (0.0-1.0)
    case_sensitive=False,
    delegate=False
)
```

### 2. ComputeExactMatch (`compute_exact_match`)
**Binary exact match accuracy** - Returns 1.0 only for perfect matches, ideal for strict evaluation where partial credit isn't appropriate

```python
operator = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_exact_match")
result = operator(
    dataset,
    pred_field="prediction",
    gt_field="ground_truth",
    output_field="exact_match",  # optional, defaults to "{pred_field}_exact_match"
    case_sensitive=False,
    strip_whitespace=True,
    delegate=False
)
```

### 3. ComputeNormalizedSimilarity (`compute_normalized_similarity`)
**Continuous similarity score** - Full range (0.0-1.0) without threshold, useful for fine-grained analysis and ranking samples by similarity

```python
operator = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_normalized_similarity")
result = operator(
    dataset,
    pred_field="prediction",
    gt_field="ground_truth",
    output_field="similarity",  # optional, defaults to "{pred_field}_similarity"
    case_sensitive=False,
    delegate=False
)
```

### 4. ComputeCER (`compute_cer`)
**Character Error Rate** - Ratio of character-level edits needed to transform prediction into ground truth (lower is better), language-agnostic

```python
operator = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_cer")
result = operator(
    dataset,
    pred_field="prediction",
    gt_field="ground_truth",
    output_field="cer",  # optional, defaults to "{pred_field}_cer"
    case_sensitive=True,
    delegate=False
)
```

### 5. ComputeWER (`compute_wer`)
**Word Error Rate** - Ratio of word-level edits needed to transform prediction into ground truth (lower is better), commonly used in speech recognition

```python
operator = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_wer")
result = operator(
    dataset,
    pred_field="prediction",
    gt_field="ground_truth",
    output_field="wer",  # optional, defaults to "{pred_field}_wer"
    case_sensitive=True,
    delegate=False
)
```

## Usage

### Via FiftyOne App

1. Open your dataset in the FiftyOne App
2. Press `` ` `` key or click the operator icon to open the Operator Browser
3. Search for the metric you want to compute (e.g., "Compute ANLS")
4. Select your prediction and ground truth StringFields
5. Configure parameters (threshold, case sensitivity, etc.)
6. Click "Execute"

The computed scores will be saved as a new field in your dataset.

### Via Python SDK

All operators support the `__call__` method for clean, Pythonic usage:

```python
import fiftyone as fo
import fiftyone.operators as foo

# Load dataset with StringFields
dataset = fo.load_dataset("your_dataset")

# Get the operator
anls_op = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_anls")

# Call operator directly - clean and simple!
result = anls_op(
    dataset,
    pred_field="prediction",
    gt_field="ground_truth",
    output_field="prediction_anls",
    threshold=0.5,
    case_sensitive=False,
)

print(f"Mean ANLS: {result['mean_anls']:.3f}")
print(f"Evaluated {result['samples_evaluated']} samples")

# View per-sample scores
print(dataset.values("prediction_anls")[:5])
```

**Smart Defaults**: The `output_field` parameter is optional and defaults to `{pred_field}_{metric}`:

```python
# These are equivalent:
result = anls_op(dataset, pred_field="prediction", gt_field="ground_truth")
# Creates field: "prediction_anls"

result = anls_op(dataset, pred_field="prediction", gt_field="ground_truth", 
                 output_field="prediction_anls")
```


### Delegated Execution

For large datasets, set `delegate=True` to run operations on a delegated service (requires delegated execution service to be running).

## Best Practices

1. **Start with ANLS**: It's the standard metric for VLM OCR tasks

2. **Use Exact Match as a secondary metric**: Provides a strict accuracy baseline

3. **Enable delegation for large datasets**: Set `delegate=True` for better performance on large datasets

4. **Organize output fields**: Use consistent prefixes (e.g., `prediction_anls`, `prediction_cer`)

5. **Evaluate on views**: Use FiftyOne's filtering to evaluate specific subsets

## Advanced Usage

### Custom Thresholds for Different Tasks

```python
# Get operator
anls_op = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_anls")

# Strict evaluation for critical fields
strict_result = anls_op(
    dataset,
    pred_field="account_number",
    gt_field="gt_account_number",
    output_field="account_anls",
    threshold=0.9,  # Higher threshold for critical data
)

# Lenient evaluation for noisy fields
lenient_result = anls_op(
    dataset,
    pred_field="description",
    gt_field="gt_description",
    output_field="description_anls",
    threshold=0.3,  # Lower threshold for descriptive text
)
```

### Comparing Multiple Models

```python
# Get operator
anls_op = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_anls")

# Evaluate two different models
models = ["model_a_prediction", "model_b_prediction"]

for model_field in models:
    result = anls_op(
        dataset,
        pred_field=model_field,
        gt_field="ground_truth",
        threshold=0.5,
    )
    print(f"{model_field}: {result['mean_anls']:.3f}")

# Compare in app
session = fo.launch_app(dataset)
```

## License

Apache 2.0

## Author

Harpreet Sahota
