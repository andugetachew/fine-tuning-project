"""
Compares aggregate stats across multiple evaluation runs (e.g. different
LoRA hyperparameters, different training runs) — same statistical rigor
as base-vs-finetuned, generalized to N named runs.
"""
import json
from pathlib import Path

from src.evaluation.stats import bootstrap_confidence_interval, paired_significance_test


def compare_runs(run_reports: dict[str, str]) -> dict:
    """run_reports: {"run_name": "path/to/base_vs_finetuned.json"}"""
    comparison = {}
    for name, path in run_reports.items():
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        scores = [r["finetuned_score"]["score"] for r in report.get("results", []) if "finetuned_score" in r]
        comparison[name] = bootstrap_confidence_interval(scores)
    return comparison


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="name=path pairs, e.g. r16=outputs/r16.json r32=outputs/r32.json")
    parser.add_argument("--output", default="outputs/evaluation/run_comparison.json")
    args = parser.parse_args()

    run_reports = dict(pair.split("=", 1) for pair in args.runs)
    result = compare_runs(run_reports)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()