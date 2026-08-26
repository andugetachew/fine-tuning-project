"""
Tests the pluggable scorer interface in isolation — no model loading,
since these should run in milliseconds as part of normal CI.
"""
import pytest
import json
from src.evaluation.scorers import get_scorer, ExactMatchScorer, NormalizedMatchScorer
from unittest.mock import patch, MagicMock

def test_exact_match_scorer():
    scorer = get_scorer("exact_match")
    assert isinstance(scorer, ExactMatchScorer)
    assert scorer.score("hello", "hello")["score"] == 1.0
    assert scorer.score("hello", "Hello")["score"] == 0.0


def test_normalized_match_scorer_ignores_case_and_punctuation():
    scorer = get_scorer("normalized_match")
    assert isinstance(scorer, NormalizedMatchScorer)
    assert scorer.score("Hello, world!", "hello world")["score"] == 1.0

def test_llm_judge_raises_without_api_key_or_package(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    scorer = get_scorer("llm_judge")
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

def test_semantic_similarity_scorer_identical_strings_near_one():
    scorer = get_scorer("semantic_similarity")
    result = scorer.score("The fix was to add a missing migration.", "The fix was to add a missing migration.")
    assert result["score"] > 0.95


def test_semantic_similarity_scorer_unrelated_strings_lower_score():
    scorer = get_scorer("semantic_similarity")
    result = scorer.score("Add a database migration for the new column.", "The weather today is sunny and warm.")
    assert result["score"] < 0.5


def test_llm_judge_scorer_parses_valid_numeric_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    scorer = get_scorer("llm_judge")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="0.8")]

    with patch("anthropic.Anthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create.return_value = mock_response
        result = scorer.score("prediction text", "gold text", prompt=[{"role": "user", "content": "context"}])

    assert result["score"] == 0.8
    assert result["details"]["metric"] == "llm_judge"


def test_llm_judge_scorer_falls_back_to_zero_on_unparseable_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    scorer = get_scorer("llm_judge")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="not a number")]

    with patch("anthropic.Anthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create.return_value = mock_response
        result = scorer.score("a", "b")

    assert result["score"] == 0.0


def test_ai_eval_framework_scorer_parses_successful_response(monkeypatch):
    scorer = get_scorer("ai_eval_framework")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"results": [{"score": 0.73}]}

    with patch("requests.post", return_value=mock_response):
        result = scorer.score("prediction", "gold")

    assert result["score"] == 0.73
    assert result["details"]["metric"] == "ai_eval_framework"

def test_compare_runs_reads_multiple_reports(tmp_path):
    from src.evaluation.compare_runs import compare_runs

    report_a = {"results": [{"finetuned_score": {"score": 0.6}}, {"finetuned_score": {"score": 0.7}}]}
    report_b = {"results": [{"finetuned_score": {"score": 0.5}}, {"finetuned_score": {"score": 0.55}}]}
    (tmp_path / "a.json").write_text(json.dumps(report_a))
    (tmp_path / "b.json").write_text(json.dumps(report_b))

    result = compare_runs({"run_a": str(tmp_path / "a.json"), "run_b": str(tmp_path / "b.json")})
    assert result["run_a"]["mean"] == pytest.approx(0.65, abs=0.01)
    assert result["run_b"]["mean"] == pytest.approx(0.525, abs=0.01)

def test_compare_runs_main_writes_output_file(tmp_path, monkeypatch, capsys):
    from src.evaluation import compare_runs as cr

    report_a = {"results": [{"finetuned_score": {"score": 0.6}}]}
    (tmp_path / "a.json").write_text(json.dumps(report_a))
    output_path = tmp_path / "comparison.json"

    monkeypatch.setattr("sys.argv", ["compare_runs.py", "--runs", f"run_a={tmp_path / 'a.json'}", "--output", str(output_path)])
    cr.main()

    assert output_path.exists()
    result = json.loads(output_path.read_text())
    assert "run_a" in result


def test_model_card_main_writes_output_file(tmp_path, monkeypatch):
    from src.evaluation import model_card as mc

    output_path = tmp_path / "CARD.md"
    monkeypatch.setattr("sys.argv", ["model_card.py", "--output", str(output_path)])
    mc.main()

    assert output_path.exists()
    content = output_path.read_text()
    assert "Qwen" in content