from .constants import DEFAULT_ALPHA, DEFAULT_TOP_K
from .config_types import SearchConfig


def default_search_config() -> SearchConfig:
    """
    Create a default retrieval configuration.
    """
    return {
        "alpha": DEFAULT_ALPHA,
        "top_k": DEFAULT_TOP_K,
    }
from .constants import DEFAULT_ALPHA, DEFAULT_TOP_K
from .config_types import SearchConfig


def create_search_config(
    alpha: float = DEFAULT_ALPHA,
    top_k: int = DEFAULT_TOP_K,
) -> SearchConfig:
    """
    Create a retrieval configuration with optional overrides.
    """
    return {
        "alpha": alpha,
        "top_k": top_k,
    }