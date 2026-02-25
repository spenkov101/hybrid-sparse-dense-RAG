def format_hybrid_scores(results, alpha: float):
    """
    Given retrieval results with dense and sparse scores,
    return a structured breakdown for inspection.

    Expected result format:
    [
        {
            "doc_id": ...,
            "dense_score": float,
            "sparse_score": float,
            "hybrid_score": float
        },
        ...
    ]
    """

    breakdown = []

    for r in results:
        dense = r.get("dense_score", 0.0)
        sparse = r.get("sparse_score", 0.0)

        hybrid = alpha * dense + (1 - alpha) * sparse

        breakdown.append({
            "doc_id": r.get("doc_id"),
            "dense_score": dense,
            "sparse_score": sparse,
            "hybrid_score": hybrid,
        })

    return breakdown
