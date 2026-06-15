from typing import TypedDict


class RetrievalResult(TypedDict):
    text: str
    score: float
    sparse_score: float
    dense_score: float