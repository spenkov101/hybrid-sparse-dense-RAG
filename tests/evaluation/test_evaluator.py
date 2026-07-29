from ir_measures import P, Recall, nDCG

from evaluation.evaluation import DEFAULT_METRICS, evaluate

def test_default_metrics_are_explicit() -> None:
    assert DEFAULT_METRICS == (
        nDCG @ 10,
        P @ 5,
        Recall @ 100,
    )


def test_evaluate_uses_default_metrics() -> None:
    qrels = {
        "q1": {
            "d1": 1,
        }
    }
    results = {
        "q1": {
            "d1": 1.0,
        }
    }

    scores = evaluate(qrels, results)

    assert set(scores) == set(DEFAULT_METRICS)
    assert scores[nDCG @ 10] == 1.0
    assert scores[P @ 5] == 0.2
    assert scores[Recall @ 100] == 1.0


def test_evaluate_accepts_custom_metrics() -> None:
    qrels = {
        "q1": {
            "d1": 1,
        }
    }
    results = {
        "q1": {
            "d1": 1.0,
        }
    }

    metric = P @ 1
    scores = evaluate(qrels, results, metrics=[metric])

    assert scores == {
        metric: 1.0,
    }