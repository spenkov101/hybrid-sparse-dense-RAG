from .constants import DEFAULT_ALPHA, DEFAULT_TOP_K
from .config_types import SearchConfig


def create_search_config(
    alpha: float = DEFAULT_ALPHA,
    top_k: int = DEFAULT_TOP_K,
) -> SearchConfig:
    """
    Create a retrieval configuration with optional overrides.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    return {
        "alpha": alpha,
        "top_k": top_k,
    }


def default_search_config() -> SearchConfig:
    return create_search_config()