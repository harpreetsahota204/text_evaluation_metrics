"""
Pure metric computation functions for text evaluation.

These functions implement standard VLM OCR evaluation metrics:
- ANLS (Average Normalized Levenshtein Similarity)
- Exact Match
- Normalized Similarity
- Character Error Rate (CER)
- Word Error Rate (WER)
"""
import Levenshtein


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate normalized Levenshtein similarity between two strings.
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Similarity score between 0.0 (completely different) and 1.0 (identical)
    """
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    
    distance = Levenshtein.distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    similarity = 1.0 - (distance / max_len)
    
    return max(0.0, similarity)


def anls(ground_truth: str, prediction: str, threshold: float = 0.5) -> float:
    """
    Calculate ANLS (Average Normalized Levenshtein Similarity) score.
    
    This is the standard metric for VLM OCR evaluation, used in DocVQA,
    OCRBench, and TextVQA benchmarks.
    
    Args:
        ground_truth: Reference text
        prediction: Model's predicted text
        threshold: Minimum similarity to count as correct (default 0.5)
    
    Returns:
        Score between 0.0 and 1.0 (below threshold returns 0.0)
    """
    similarity = normalized_levenshtein_similarity(
        ground_truth.strip(), 
        prediction.strip()
    )
    
    return similarity if similarity >= threshold else 0.0


def exact_match(ground_truth: str, prediction: str, 
                case_sensitive: bool = False,
                strip_whitespace: bool = True) -> float:
    """
    Check if prediction exactly matches ground truth.
    
    Args:
        ground_truth: Reference text
        prediction: Model's predicted text
        case_sensitive: Whether to use case-sensitive comparison
        strip_whitespace: Whether to strip leading/trailing whitespace
    
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    gt = ground_truth.strip() if strip_whitespace else ground_truth
    pred = prediction.strip() if strip_whitespace else prediction
    
    if not case_sensitive:
        gt = gt.lower()
        pred = pred.lower()
    
    return 1.0 if gt == pred else 0.0


def character_error_rate(ground_truth: str, prediction: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    Traditional OCR metric measuring character-level edit distance.
    
    Args:
        ground_truth: Reference text
        prediction: Model's predicted text
    
    Returns:
        CER value (0.0 = perfect, higher = more errors)
    """
    if len(ground_truth) == 0 and len(prediction) == 0:
        return 0.0
    if len(ground_truth) == 0:
        return 1.0
    
    distance = Levenshtein.distance(ground_truth, prediction)
    return distance / len(ground_truth)


def word_error_rate(ground_truth: str, prediction: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    Traditional OCR metric measuring word-level edit distance.
    
    Args:
        ground_truth: Reference text
        prediction: Model's predicted text
    
    Returns:
        WER value (0.0 = perfect, higher = more errors)
    """
    gt_words = ground_truth.split()
    pred_words = prediction.split()
    
    if len(gt_words) == 0 and len(pred_words) == 0:
        return 0.0
    if len(gt_words) == 0:
        return 1.0
    
    distance = Levenshtein.distance(gt_words, pred_words)
    return distance / len(gt_words)

