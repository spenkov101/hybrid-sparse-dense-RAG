def print_top_results(results, top_k=3):
    """
    Pretty-print top retrieval results.

    Args:
        results (list of dict):
            Retrieval results from HybridRetriever.
        top_k (int):
            Number of top results to display.
    """
    for i, result in enumerate(results[:top_k], start=1):
        print(f"\nRank #{i}")
        print(f"Score: {result['score']:.4f}")

        if "sparse_score" in result:
            print(f"Sparse: {result['sparse_score']:.4f}")

        if "dense_score" in result:
            print(f"Dense: {result['dense_score']:.4f}")

        print("Text:")
        print(result["text"])
        print("-" * 50)