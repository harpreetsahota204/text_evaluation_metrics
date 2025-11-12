"""
Test script for text evaluation metrics.

Run this to validate the implementation.
"""

import fiftyone as fo


def test_text_metrics():
    """Test text evaluation metrics with sample data."""
    
    # Create test dataset
    dataset = fo.Dataset("text_eval_test", overwrite=True)
    
    test_cases = [
        ("Invoice #12345", "Invoice #12345"),      # Perfect match
        ("Invoice #12345", "Invoice 12345"),       # High similarity
        ("$1,234.56", "$1234.56"),                 # Medium similarity
        ("January 15, 2024", "Jan 15, 2024"),      # Lower similarity
        ("Hello World", "Goodbye Moon"),           # Very low similarity
    ]
    
    samples = [
        fo.Sample(
            filepath=f"test_{i}.txt",
            ground_truth=gt,
            prediction=pred,
        )
        for i, (gt, pred) in enumerate(test_cases)
    ]
    
    dataset.add_samples(samples)
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Run evaluation with all metrics
    results = dataset.evaluate_regressions(
        "prediction",
        gt_field="ground_truth",
        eval_key="eval",
        method="text",
        custom_metrics=[
            "@harpreetsahota/text-evaluation-metrics/anls",
            "@harpreetsahota/text-evaluation-metrics/exact_match",
            "@harpreetsahota/text-evaluation-metrics/normalized_similarity",
            "@harpreetsahota/text-evaluation-metrics/cer",
            "@harpreetsahota/text-evaluation-metrics/wer",
        ],
    )
    
    # Print results
    print("\nAggregate Metrics:")
    print(results.metrics())
    
    print("\nPer-sample ANLS scores:")
    print(dataset.values("eval_anls"))
    
    print("\nPer-sample Exact Match scores:")
    print(dataset.values("eval_exact_match"))
    
    # Verify fields were created
    schema = dataset.get_field_schema()
    assert "eval_anls" in schema
    assert "eval_exact_match" in schema
    assert "eval_normalized_similarity" in schema
    assert "eval_cer" in schema
    assert "eval_wer" in schema
    print("\n✓ All metric fields created successfully")
    
    # Test cleanup
    dataset.delete_evaluation("eval")
    schema = dataset.get_field_schema()
    assert "eval_anls" not in schema
    print("✓ Cleanup successful")
    
    # Cleanup dataset
    dataset.delete()
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_text_metrics()

