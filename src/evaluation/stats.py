"""
Statistical rigor for evaluation results: bootstrap confidence intervals
and significance testing when comparing base vs fine-tuned model scores.
Operates on an existing base_vs_finetuned.json report - no model loading
or GPU required.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


def bootstrap_confidence_interval(scores: list[float], n_resamples: int = 10000, confidence: float = 0.95) -> dict:
    """Bootstrap resampling to estimate a confidence interval around the mean score."""
    if not scores:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    scores_arr = np.array(scores)
    rng = np.random.default_rng(seed=42)
    means = [rng.choice(scores_arr, size=len(scores_arr), replace=True).mean() for _ in range(n_resamples)]
    alpha = 1 - confidence
    ci_low = float(np.percentile(means, 100 * alpha / 2))
    ci_high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {"mean": float(scores_arr.mean()), "ci_low": ci_low, "ci_high": ci_high, "n": len(scores)}


def paired_significance_test(base_scores: list[float], finetuned_scores: list[float]) -> dict:
    """Wilcoxon signed-rank test - appropriate for small, paired, non-normal
    samples like a handful of eval examples scored by both models."""
    if len(base_scores) != len(finetuned_scores) or len(base_scores) < 2:
        return {"test": "wilcoxon", "p_value": None, "significant": None, "note": "insufficient paired samples"}
    try:
        statistic, p_value = scipy_stats.wilcoxon(base_scores, finetuned_scores)
        return {
            "test": "wilcoxon",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "note": "p < 0.05 suggests the difference is unlikely due to chance, "
                    "though a sample this small has low statistical power",
        }
    except ValueError as e:
        return {"test": "wilcoxon", "p_value": None, "significant": None, "note": str(e)}


def aggregate_comparison_report(report_path: str) -> dict:
    """Reads a base_vs_finetuned.json comparison report and computes aggregate
    statistics across all eval examples, rather than only reporting per-example
    scores."""
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    results = report.get("results", [])
    base_scores = [r["base_score"]["score"] for r in results if "base_score" in r]
    finetuned_scores = [r["finetuned_score"]["score"] for r in results if "finetuned_score" in r]

    return {
        "scorer_type": report.get("scorer_type"),
        "num_examples": len(results),
        "num_scored": len(base_scores),
        "base": bootstrap_confidence_interval(base_scores),
        "finetuned": bootstrap_confidence_interval(finetuned_scores),
        "significance": paired_significance_test(base_scores, finetuned_scores),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="outputs/evaluation/base_vs_finetuned.json")
    parser.add_argument("--output", default="outputs/evaluation/aggregate_stats.json")
    args = parser.parse_args()

    aggregate = aggregate_comparison_report(args.report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(json.dumps(aggregate, indent=2))
    print(f"\nAggregate stats written to {out_path}")


if __name__ == "__main__":
    main()