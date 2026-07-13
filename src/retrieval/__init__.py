"""
Retrieval components for hybrid sparse-dense search.

Exports:

- DenseRetriever
- SpladeRetriever
- HybridRetriever
- default_search_config
- create_search_config
- get_top_result
"""

from .dense import DenseRetriever
from .splade import SpladeRetriever
from .hybrid_retriever import HybridRetriever
from .config_utils import default_search_config, create_search_config
from .result_utils import get_top_result

__all__ = [
    "DenseRetriever",
    "SpladeRetriever",
    "HybridRetriever",
    "default_search_config",
    "create_search_config",
    "get_top_result",
]