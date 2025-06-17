import ir_measures
from ir_measures import nDCG, P, Recall, MAP

def evaluate(qrels: dict, results: dict, metrics: list = [nDCG@10, P@5, Recall@100]):
    """BEIR evaluation wrapper"""
    return ir_measures.calc_aggregate(metrics, qrels, results)
