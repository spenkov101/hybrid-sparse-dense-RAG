# Retrieval Public API

The retrieval package exposes the main user-facing components from `src.retrieval`.

Prefer importing from the package-level API:

```python
from src.retrieval import (
    HybridRetriever,
    create_search_config,
    get_top_result,
)