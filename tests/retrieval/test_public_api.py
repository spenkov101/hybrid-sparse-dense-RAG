import src.retrieval as retrieval


EXPECTED_PUBLIC_API = {
    "DenseRetriever",
    "SpladeRetriever",
    "HybridRetriever",
    "default_search_config",
    "create_search_config",
    "get_top_result",
}


def test_retrieval_public_api_exports_expected_names() -> None:
    assert EXPECTED_PUBLIC_API.issubset(set(retrieval.__all__))

    for name in EXPECTED_PUBLIC_API:
        assert hasattr(retrieval, name)