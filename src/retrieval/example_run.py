from retrieval.hybrid_retriever import HybridRetriever

def demo():
    query = "What is the French capital?"
    passages = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Sofia is the capital of Bulgaria."
    ]

    retriever = HybridRetriever(alpha=0.5)
    results = retriever.search(query, passages)

    print(f"Query: {query}")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['text']} (score={r['score']:.4f})")

if __name__ == "__main__":
    demo()
