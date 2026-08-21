"""
Tests the pluggable scorer interface in isolation — no model loading,
since these should run in milliseconds as part of normal CI.
"""
import pytest

from src.evaluation.scorers import get_scorer, ExactMatchScorer, NormalizedMatchScorer


def test_exact_match_scorer():
    scorer = get_scorer("exact_match")
    assert isinstance(scorer, ExactMatchScorer)
    assert scorer.score("hello", "hello")["score"] == 1.0
    assert scorer.score("hello", "Hello")["score"] == 0.0


def test_normalized_match_scorer_ignores_case_and_punctuation():
    scorer = get_scorer("normalized_match")
    assert isinstance(scorer, NormalizedMatchScorer)
    assert scorer.score("Hello, world!", "hello world")["score"] == 1.0


def test_unimplemented_scorers_raise_not_implemented():
    for scorer_type in ["llm_judge", "ai_eval_framework"]:
        scorer = get_scorer(scorer_type)
        with pytest.raises(NotImplementedError):
            scorer.score("a", "b")


def test_unknown_scorer_type_raises_value_error():
    with pytest.raises(ValueError):
        get_scorer("not_a_real_scorer")
def test_llm_judge_scorer_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    scorer = get_scorer("llm_judge")
    with pytest.raises(NotImplementedError):
        scorer.score("a", "b")

def test_ai_eval_framework_scorer_raises_on_connection_failure(monkeypatch):
    monkeypatch.setenv("AI_EVAL_FRAMEWORK_URL", "http://localhost:1")  # unreachable port
    scorer = get_scorer("ai_eval_framework")
    with pytest.raises(NotImplementedError):
        scorer.score("a", "b")