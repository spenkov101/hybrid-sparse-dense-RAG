"""
BEIR-style evaluation using ir_measures.

This module exposes a simple `evaluate` function that computes
standard information-retrieval metrics over qrels and ranked results.
"""

import ir_measures
from ir_measures import P, Recall, nDCG


DEFAULT_METRICS = (
    nDCG @ 10,
    P @ 5,
    Recall @ 100,
)


def evaluate(qrels: dict, results: dict, metrics=None):
    """
    Evaluate retrieval performance.

    Args:
        qrels: Ground-truth relevance judgements.
        results: Retrieved documents and their scores.
        metrics: Metrics to compute. Uses DEFAULT_METRICS when omitted.

    Returns:
        Aggregate metric scores produced by ir_measures.
    """
    selected_metrics = DEFAULT_METRICS if metrics is None else metrics

    return ir_measures.calc_aggregate(
        selected_metrics,
        qrels,
        results,
    )