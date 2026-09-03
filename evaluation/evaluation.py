"""Provide evaluation utilities for retrieval experiments."""

import ir_measures
from ir_measures import P, Recall, nDCG


DEFAULT_METRICS = (
    nDCG @ 10,
    P @ 5,
    Recall @ 100,
)


def evaluate(
    qrels: dict,
    results: dict,
    metrics=None,
) -> dict:
    """
    Evaluate ranked retrieval results.

    :param qrels: Ground-truth relevance judgments.
    :param results: Retrieved documents and their scores.
    :param metrics: Optional metrics to compute.
    :return: Aggregate evaluation scores keyed by metric.
    """
    selected_metrics = DEFAULT_METRICS if metrics is None else metrics

    return ir_measures.calc_aggregate(
        selected_metrics,
        qrels,
        results,
    )


def run_beir_evaluation(
    qrels: dict,
    results: dict,
    metrics=None,
) -> dict:
    """
    Run retrieval evaluation through the high-level interface.

    :param qrels: Ground-truth relevance judgments.
    :param results: Retrieved documents and their scores.
    :param metrics: Optional metrics to compute.
    :return: Aggregate evaluation scores keyed by metric.
    """
    return evaluate(qrels, results, metrics)

def serialize_evaluation_results(scores: dict) -> dict[str, float]:
    """
    Convert evaluation results to a serializable dictionary.

    :param scores: Evaluation scores keyed by metric objects.
    :return: Evaluation scores keyed by metric names.
    """
    return {
        str(metric): float(score)
        for metric, score in scores.items()
    }