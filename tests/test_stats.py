import json
import pytest

from src.evaluation.stats import bootstrap_confidence_interval, paired_significance_test, aggregate_comparison_report
from src.evaluation.model_card import generate_model_card


def test_bootstrap_confidence_interval_basic():
    result = bootstrap_confidence_interval([0.5, 0.6, 0.55, 0.52, 0.58], n_resamples=1000)
    assert result["n"] == 5
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_bootstrap_confidence_interval_empty_list():
    result = bootstrap_confidence_interval([])
    assert result["n"] == 0
    assert result["mean"] is None


def test_paired_significance_test_insufficient_samples():
    result = paired_significance_test([0.5], [0.6])
    assert result["p_value"] is None
    assert "insufficient" in result["note"]


def test_paired_significance_test_returns_wilcoxon_result():
    result = paired_significance_test([0.4, 0.5, 0.3, 0.6], [0.6, 0.55, 0.7, 0.65])
    assert result["test"] == "wilcoxon"
    assert isinstance(result["p_value"], float)


def test_aggregate_comparison_report_reads_real_report_shape(tmp_path):
    report = {
        "scorer_type": "semantic_similarity",
        "results": [
            {"base_score": {"score": 0.4}, "finetuned_score": {"score": 0.6}},
            {"base_score": {"score": 0.5}, "finetuned_score": {"score": 0.55}},
            {"base_score": {"score": 0.3}, "finetuned_score": {"score": 0.7}},
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = aggregate_comparison_report(str(report_path))
    assert result["num_examples"] == 3
    assert result["num_scored"] == 3
    assert result["base"]["mean"] == pytest.approx(0.4, abs=0.01)
    assert result["finetuned"]["mean"] == pytest.approx(0.6167, abs=0.01)


def test_generate_model_card_includes_base_model_id():
    card = generate_model_card()
    assert "Qwen/Qwen2.5-1.5B-Instruct" in card
    assert "LoRA" in card
    assert "Known Limitations" in card

def test_stats_main_writes_output_file(tmp_path, monkeypatch):
    from src.evaluation import stats as st

    report = {
        "scorer_type": "semantic_similarity",
        "results": [
            {"base_score": {"score": 0.4}, "finetuned_score": {"score": 0.6}},
            {"base_score": {"score": 0.5}, "finetuned_score": {"score": 0.55}},
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    output_path = tmp_path / "aggregate.json"

    monkeypatch.setattr("sys.argv", ["stats.py", "--report", str(report_path), "--output", str(output_path)])
    st.main()

    assert output_path.exists()