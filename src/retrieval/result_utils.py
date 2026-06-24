from .result_types import RetrievalResult


def get_top_result(results: list[RetrievalResult]) -> RetrievalResult | None:
    """
    Return the highest-ranked retrieval result.

    Returns None if no results are available.
    """
    if not results:
        return None

    return results[0]