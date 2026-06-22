# Retrieval Configuration

The retrieval pipeline uses a small default configuration helper for sanity checks and exploratory experiments.

Current defaults:

- `alpha`: controls the balance between sparse and dense retrieval scores
- `top_k`: controls how many ranked results are returned

The default configuration can be loaded with:

```python
from src.retrieval import default_search_config

config = default_search_config()