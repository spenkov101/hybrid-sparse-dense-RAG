from evaluation.beir_metrics.evaluator import evaluate

def run_beir_evaluation(qrels, results):
    """Call this from your main evaluation flow"""
    return evaluate(qrels, results)
