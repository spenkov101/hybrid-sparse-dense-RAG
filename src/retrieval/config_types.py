from typing import TypedDict


class SearchConfig(TypedDict):
    alpha: float
    top_k: int