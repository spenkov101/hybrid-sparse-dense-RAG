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
