"""
Evaluates base model vs LoRA-fine-tuned model on the same eval set,
scores both with the configured scorer, and writes a comparison
report to outputs/evaluation/.

Usage:
    python -m src.evaluation.evaluate --adapter models/checkpoints/final_adapter
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset

from src.training.model_utils import load_config, load_base_model_and_tokenizer, load_finetuned_model
from src.evaluation.scorers import get_scorer


def generate_response(model, tokenizer, messages: list[dict], max_new_tokens: int = 256) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_model_on_eval_set(model, tokenizer, eval_examples: list[dict]) -> list[dict]:
    results = []
    for ex in eval_examples:
        messages = ex["messages"][:-1]
        gold = ex["messages"][-1]["content"]
        prediction = generate_response(model, tokenizer, messages)
        results.append({"prompt": messages, "gold": gold, "prediction": prediction})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    eval_examples = list(load_dataset("json", data_files=config["dataset"]["eval_path"], split="train"))

    print("Running base model...")
    base_model, base_tokenizer = load_base_model_and_tokenizer(config)
    base_results = run_model_on_eval_set(base_model, base_tokenizer, eval_examples)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Running fine-tuned model...")
    ft_model, ft_tokenizer = load_finetuned_model(
        config["model"]["base_model_id"], args.adapter, config["model"]["trust_remote_code"]
    )
    ft_results = run_model_on_eval_set(ft_model, ft_tokenizer, eval_examples)

    scorer_type = config["evaluation"]["scorer_type"]
    scorer = get_scorer(scorer_type)

    scored_pairs = []
    scoring_skipped_reason = None
    for base_r, ft_r in zip(base_results, ft_results):
        gold = base_r["gold"]
        pair = {
            "prompt": base_r["prompt"],
            "gold": gold,
            "base_prediction": base_r["prediction"],
            "finetuned_prediction": ft_r["prediction"],
        }
        try:
            pair["base_score"] = scorer.score(base_r["prediction"], gold)
            pair["finetuned_score"] = scorer.score(ft_r["prediction"], gold)
        except NotImplementedError as e:
            scoring_skipped_reason = str(e)
        scored_pairs.append(pair)

    comparison = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model_id": config["model"]["base_model_id"],
        "adapter_path": args.adapter,
        "scorer_type": scorer_type,
        "num_eval_examples": len(eval_examples),
        "scoring_skipped_reason": scoring_skipped_reason,
        "results": scored_pairs,
    }

    out_path = Path("outputs/evaluation/base_vs_finetuned.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"Comparison report written to {out_path}")


if __name__ == "__main__":
    main()
