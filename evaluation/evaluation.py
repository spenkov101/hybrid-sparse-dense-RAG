"""
High-level evaluation interface for hybrid/dense/sparse retrieval.

This module wraps BEIR-style evaluation using ir_measures.
Use `run_beir_evaluation` inside experiments to compute
nDCG, MAP, Recall, Precision, etc.
"""

from evaluation.beir_metrics.evaluator import evaluate


def run_beir_evaluation(qrels, results, metrics=None):
    """
    Run BEIR/IR-style evaluation over ranked retrieval results.

    Args:
        qrels (dict): Ground truth relevance judgments.
        results (dict): Retrieved documents with scores.
        metrics (list, optional): List of metrics (default inside evaluator).

    Returns:
        dict: Aggregate evaluation scores.
    """
    return evaluate(qrels, results, metrics)
