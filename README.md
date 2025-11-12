# Text Evaluation Metrics for FiftyOne

Operator plugin for evaluating text fields (StringFields) in FiftyOne datasets with standard VLM OCR metrics.

## Features

- **Modular Design**: Each metric is a separate operator
- **Flexible**: Choose exactly which metrics you need
- **Intelligent Defaults**: Smart field name suggestions
- **Delegated Execution**: Automatic delegation for large datasets (>1000 samples)
- **Clean Interface**: Simple, focused UI for each metric

## Installation

```bash
pip install python-Levenshtein
```

## Available Operators

### 1. Compute ANLS
**Average Normalized Levenshtein Similarity** - Primary metric for VLM OCR evaluation

- Configurable threshold (default: 0.5)
- Case-sensitive option
- Standard metric used in OCR benchmarks

### 2. Compute Exact Match
**Binary exact match accuracy** between prediction and ground truth

- Case-sensitive option
- Whitespace stripping option
- Returns 1.0 for perfect match, 0.0 otherwise

### 3. Compute Normalized Similarity
**Continuous similarity score** (0.0-1.0) without threshold

- No threshold applied
- Full range of similarity values
- Useful for ranking and analysis

### 4. Compute CER
**Character Error Rate** - Ratio of character edits needed

- Based on Levenshtein distance at character level
- Lower is better (0.0 = perfect)
- Case-sensitive by default

### 5. Compute WER
**Word Error Rate** - Ratio of word edits needed

- Based on Levenshtein distance at word level
- Lower is better (0.0 = perfect)
- Case-sensitive by default

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

### Programmatic Batch Evaluation

Run multiple metrics at once using the clean `__call__` interface:

```python
import fiftyone as fo
import fiftyone.operators as foo

dataset = fo.load_dataset("your_dataset")

# Get operators
anls_op = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_anls")
em_op = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_exact_match")
cer_op = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_cer")
wer_op = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_wer")

# Call operators directly - clean and readable!
results = {
    "anls": anls_op(
        dataset,
        pred_field="prediction",
        gt_field="ground_truth",
        threshold=0.5,
    ),
    "exact_match": em_op(
        dataset,
        pred_field="prediction",
        gt_field="ground_truth",
    ),
    "cer": cer_op(
        dataset,
        pred_field="prediction",
        gt_field="ground_truth",
    ),
    "wer": wer_op(
        dataset,
        pred_field="prediction",
        gt_field="ground_truth",
    ),
}

# Print summary
print(f"ANLS: {results['anls']['mean_anls']:.3f}")
print(f"Exact Match: {results['exact_match']['accuracy']:.3f}")
print(f"CER: {results['cer']['mean_cer']:.3f}")
print(f"WER: {results['wer']['mean_wer']:.3f}")
```

### Working with Views

The operators work with FiftyOne views for targeted evaluation:

```python
import fiftyone as fo
import fiftyone.operators as foo

dataset = fo.load_dataset("your_dataset")

# Get operator
anls_op = foo.get_operator("@harpreetsahota/text-evaluation-metrics/compute_anls")

# Create a view (e.g., only samples with confidence > 0.5)
high_confidence_view = dataset.match(fo.F("confidence") > 0.5)

# Execute operator on view using __call__
result = anls_op(
    high_confidence_view,
    pred_field="prediction",
    gt_field="ground_truth",
    output_field="anls_high_confidence",
    threshold=0.5,
)

print(f"High-confidence samples ANLS: {result['mean_anls']:.3f}")
```

### Delegated Execution

For large datasets or long-running operations, request delegated execution:

```python
# Request delegation (runs in background)
result = anls_op(
    dataset,
    pred_field="prediction",
    gt_field="ground_truth",
    delegate=True,  # Request background execution
)

# Note: Automatic delegation happens for datasets > 1000 samples
```

## Example Workflow

```python
import fiftyone as fo

# Create sample dataset
dataset = fo.Dataset("text_eval_demo")

samples = [
    fo.Sample(
        filepath="invoice_1.jpg",
        ground_truth="Invoice #12345",
        prediction="Invoice 12345"  # Missing '#'
    ),
    fo.Sample(
        filepath="invoice_2.jpg",
        ground_truth="Total: $1,234.56",
        prediction="Total: $1,234.56"  # Perfect match
    ),
    fo.Sample(
        filepath="receipt_1.jpg",
        ground_truth="Thank you for your purchase",
        prediction="Thank you for youre purchase"  # Typo
    ),
]

dataset.add_samples(samples)

# Launch app
session = fo.launch_app(dataset)

# Use the Operator Browser to compute metrics
# Press ` key, search "Compute ANLS", fill in fields, execute
```

## Metric Descriptions

### ANLS (Average Normalized Levenshtein Similarity)

ANLS is the standard metric for OCR evaluation in VLMs. It:
- Normalizes edit distance by string length
- Applies a threshold (typically 0.5)
- Returns 1.0 if similarity ≥ threshold, otherwise returns the similarity score
- Is robust to minor OCR errors

**Use case**: Primary evaluation metric for OCR tasks, VLM document understanding

### Exact Match

Binary metric that returns 1.0 only for perfect matches.

**Use case**: Strict evaluation where partial credit isn't appropriate (e.g., form field extraction)

### Normalized Similarity

Continuous similarity without threshold, ranging from 0.0 to 1.0.

**Use case**: Fine-grained analysis, ranking samples by similarity

### CER (Character Error Rate)

Ratio of character-level edits needed to transform prediction into ground truth.

**Use case**: Detailed character-level error analysis, language-agnostic evaluation

### WER (Word Error Rate)

Ratio of word-level edits needed to transform prediction into ground truth.

**Use case**: Speech recognition, word-level accuracy analysis

## Field Management

### Viewing Computed Fields

```python
# List all fields
print(dataset.get_field_schema())

# View specific metric values
print(dataset.values("anls"))
print(dataset.values("exact_match"))

# Statistics
print(dataset.mean("anls"))
print(dataset.std("anls"))
```

### Deleting Fields

```python
# Delete specific fields
dataset.delete_sample_fields(["anls", "exact_match", "cer"])

# Or delete all evaluation fields at once
eval_fields = [f for f in dataset.get_field_schema().keys() if f.startswith("eval_")]
dataset.delete_sample_fields(eval_fields)
```

## Best Practices

1. **Start with ANLS**: It's the standard metric for VLM OCR tasks
2. **Use Exact Match as a secondary metric**: Provides a strict accuracy baseline
3. **Enable delegation for large datasets**: Operators auto-delegate for >1000 samples
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

## Troubleshooting

### "No StringFields Found" Warning

Ensure your dataset has StringFields:

```python
# Add StringFields if needed
dataset.add_sample_field("prediction", fo.StringField)
dataset.add_sample_field("ground_truth", fo.StringField)
```

### Import Error: "No module named 'Levenshtein'"

Install the required dependency:

```bash
pip install python-Levenshtein
```

### Performance Issues

For large datasets (>10,000 samples), the operator automatically delegates execution to a background process. You can also manually create batches:

```python
# Process in chunks
batch_size = 5000
for i in range(0, len(dataset), batch_size):
    batch_view = dataset.skip(i).limit(batch_size)
    result = anls_op.execute(batch_view, params={...})
```

## License

Apache 2.0

## Author

Harpreet Sahota
