def pretty_print_results(results, top_k: int = 5):
    """
    Nicely print retrieval results for inspection.

    Expected format:
    [
        {
            "doc_id": str,
            "text": str,
            "score": float
        }
    ]
    """

    print("\n=== Retrieval Results ===\n")

    for i, r in enumerate(results[:top_k], start=1):
        print(f"[{i}] Doc ID: {r.get('doc_id')}")
        print(f"Score : {r.get('score'):.4f}")
        text = r.get("text", "")
        print(f"Text  : {text[:200]}...")
        print("-" * 50)
def retrieval_overlap(dense_results, sparse_results, top_k: int = 10):
    """
    Compute overlap between dense and sparse retrieval results.

    Parameters
    ----------
    dense_results : list[dict]
    sparse_results : list[dict]

    Each result must contain:
        {"doc_id": ...}

    Returns
    -------
    dict with overlap statistics.
    """

    dense_ids = {
        r["doc_id"] for r in dense_results[:top_k]
        if "doc_id" in r
    }

    sparse_ids = {
        r["doc_id"] for r in sparse_results[:top_k]
        if "doc_id" in r
    }

    intersection = dense_ids & sparse_ids
    union = dense_ids | sparse_ids

    return {
        "dense_top_k": len(dense_ids),
        "sparse_top_k": len(sparse_ids),
        "overlap": len(intersection),
        "overlap_ratio": len(intersection) / len(union) if union else 0.0,
    }

def retrieval_difference(dense_results, sparse_results, top_k: int = 10):
    """
    Identify documents unique to dense or sparse retrieval.

    Returns:
        {
            "dense_only": set,
            "sparse_only": set
        }
    """

    dense_ids = {
        r["doc_id"] for r in dense_results[:top_k]
        if "doc_id" in r
    }

    sparse_ids = {
        r["doc_id"] for r in sparse_results[:top_k]
        if "doc_id" in r
    }

    return {
        "dense_only": dense_ids - sparse_ids,
        "sparse_only": sparse_ids - dense_ids,
    }
