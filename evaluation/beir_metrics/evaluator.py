"""
BEIR-style evaluation using ir_measures.

This module exposes a simple `evaluate` function that computes
standard IR metrics over (qrel, result) pairs.
"""

import ir_measures
from ir_measures import nDCG, P, Recall, MAP


def evaluate(qrels: dict, results: dict, metrics: list = None):
    """
    Evaluate retrieval performance.

    Args:
        qrels (dict): Ground truth mapping qid → {doc_id: relevance}
        results (dict): System results mapping qid → list of (doc_id, score)
        metrics (list): List of IR metrics to compute.
                        Defaults to [nDCG@10, P@5, Recall@100].

    Returns:
        dict: Aggregate metric scores.
    """
    if metrics is None:
        metrics = [nDCG@10, P@5, Recall@100]

    return ir_measures.calc_aggregate(metrics, qrels, results)

