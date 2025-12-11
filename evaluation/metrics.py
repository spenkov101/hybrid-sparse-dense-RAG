"""
Evaluation utilities for retrieval experiments.

This module defines a minimal scoring interface for hybrid, dense,
and sparse retrieval results. Metrics will be implemented later.
"""

from typing import List, Dict, Any

def compute_ranked_metrics(results: List[Dict[str, Any]], k: int = 5) -> Dict[str, float]:
    """
    Placeholder for retrieval metrics (MAP, nDCG, Recall).
    
    Args:
        results: A ranked list of dicts containing at least 'score' and 'text'
        k: Cutoff for metrics like MAP@k or nDCG@k.
    
    Returns:
        Dict with metric names and placeholder values.
    """
    return {
        f"MAP@{k}": None,
        f"nDCG@{k}": None,
        f"Recall@{k}": None,
    }


if __name__ == "__main__":
    # Tiny self-test to confirm the module loads
    mock = [{"text": "example", "score": 0.9}]
    print(compute_ranked_metrics(mock))
