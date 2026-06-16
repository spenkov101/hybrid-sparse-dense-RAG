from typing import TypedDict


class EvaluationResult(TypedDict):
    recall_at_k: float
    ndcg_at_k: float
    map_score: float
    mrr: float