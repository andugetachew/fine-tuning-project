"""
Generates a model card summarizing the fine-tuning run: base model,
LoRA config, dataset size, and known limitations. Standard practice for
documenting a trained model's provenance.
"""
from pathlib import Path

from src.training.model_utils import load_config


TEMPLATE = """# Model Card: {model_name}

## Base Model
- **Base model**: `{base_model_id}`
- **Adapter type**: LoRA (r={lora_r}, alpha={lora_alpha}, target_modules={target_modules})

## Training Data
- **Examples**: {num_examples} real, verified backend engineering incidents
- **Domain**: {domain}
- **Train/eval split**: {train_count} train / {eval_count} eval

## Training
- **Epochs**: {epochs}
- **Batch size**: {batch_size} (effective: {effective_batch_size} with gradient accumulation)
- **Learning rate**: {learning_rate}

## Evaluation
- **Scorer**: {scorer_type}
- **Result**: see `outputs/evaluation/aggregate_stats.json` for mean scores,
  confidence intervals, and significance testing.

## Known Limitations
- Small training set ({num_examples} examples) — not expected to produce a
  large, statistically robust capability shift.
- Eval set is small ({eval_count} examples) — confidence intervals are wide;
  treat results as directional, not conclusive.
- `llm_judge` and `ai_eval_framework` scorers depend on external services
  and won't run without credentials/connectivity.

## Intended Use
A portfolio/demonstration project showing an end-to-end QLoRA fine-tuning
pipeline: data curation, training, evaluation, and serving. Not intended
for production use without a substantially larger, more rigorously
validated dataset.
"""


def generate_model_card(config_path: str = "configs/qlora.yaml") -> str:
    config = load_config(config_path)
    lora = config["lora"]
    training = config["training"]

    return TEMPLATE.format(
        model_name="Backend Engineering Assistant",
        base_model_id=config["model"]["base_model_id"],
        lora_r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        target_modules=", ".join(lora["target_modules"]),
        num_examples="24",
        domain="Django/DRF, PostgreSQL, Docker, async systems, Python internals, ML pipelines",
        train_count="21",
        eval_count="3",
        epochs=training["num_train_epochs"],
        batch_size=training["per_device_train_batch_size"],
        effective_batch_size=training["per_device_train_batch_size"] * training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        scorer_type=config["evaluation"]["scorer_type"],
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--output", default="MODEL_CARD.md")
    args = parser.parse_args()

    card = generate_model_card(args.config)
    Path(args.output).write_text(card, encoding="utf-8")
    print(f"Model card written to {args.output}")


if __name__ == "__main__":
    main()