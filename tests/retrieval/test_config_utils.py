import pytest

from src.retrieval import create_search_config, default_search_config
from src.retrieval.constants import DEFAULT_ALPHA, DEFAULT_TOP_K


def test_create_search_config_uses_defaults() -> None:
    config = create_search_config()

    assert config["alpha"] == DEFAULT_ALPHA
    assert config["top_k"] == DEFAULT_TOP_K


def test_create_search_config_accepts_overrides() -> None:
    config = create_search_config(alpha=0.7, top_k=3)

    assert config["alpha"] == 0.7
    assert config["top_k"] == 3


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_create_search_config_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        create_search_config(alpha=alpha)


@pytest.mark.parametrize("top_k", [0, -1])
def test_create_search_config_rejects_invalid_top_k(top_k: int) -> None:
    with pytest.raises(ValueError):
        create_search_config(top_k=top_k)


def test_default_search_config_matches_default_values() -> None:
    config = default_search_config()

    assert config == {
        "alpha": DEFAULT_ALPHA,
        "top_k": DEFAULT_TOP_K,
    }