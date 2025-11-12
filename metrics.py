"""
Text evaluation metric computation functions.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import Levenshtein


def normalized_levenshtein_similarity(s1, s2, case_sensitive=False):
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


def compute_anls(gt, pred, threshold=0.5, case_sensitive=False):
    """Compute ANLS score with threshold."""
    similarity = normalized_levenshtein_similarity(
        gt.strip(), 
        pred.strip(), 
        case_sensitive
    )
    return similarity if similarity >= threshold else 0.0


def compute_exact_match(gt, pred, case_sensitive=False, strip_whitespace=True):
    """Compute exact match score (1.0 or 0.0)."""
    gt_norm = gt.strip() if strip_whitespace else gt
    pred_norm = pred.strip() if strip_whitespace else pred
    
    if not case_sensitive:
        gt_norm = gt_norm.lower()
        pred_norm = pred_norm.lower()
    
    return 1.0 if gt_norm == pred_norm else 0.0


def compute_normalized_similarity(gt, pred, case_sensitive=False):
    """Compute normalized similarity without threshold."""
    return normalized_levenshtein_similarity(gt.strip(), pred.strip(), case_sensitive)


def compute_cer(gt, pred, case_sensitive=True):
    """Compute Character Error Rate."""
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    
    gt_norm = gt if case_sensitive else gt.lower()
    pred_norm = pred if case_sensitive else pred.lower()
    
    distance = Levenshtein.distance(gt_norm, pred_norm)
    return distance / len(gt)


def compute_wer(gt, pred, case_sensitive=True):
    """Compute Word Error Rate."""
    gt_words = gt.split()
    pred_words = pred.split()
    
    if len(gt_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    
    if not case_sensitive:
        gt_words = [w.lower() for w in gt_words]
        pred_words = [w.lower() for w in pred_words]
    
    distance = Levenshtein.distance(gt_words, pred_words)
    return distance / len(gt_words)


def safe_mean(values):
    """Compute mean, filtering out None values."""
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None

