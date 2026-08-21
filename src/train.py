"""
CLI entry point for QLoRA fine-tuning. All real logic lives in
src/training/ — this file just wires config -> model -> trainer -> run.

Usage:
    python -m src.train --config configs/qlora.yaml
    python -m src.train --config configs/qlora.yaml --dry-run
"""
import argparse
import os
import torch

torch.set_num_threads(os.cpu_count())

from src.training.model_utils import load_config, load_base_model_and_tokenizer
from src.training.trainer import build_datasets, build_trainer, prepare_model_for_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 epoch on a 5-example slice — local CPU sanity check before Colab.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model, tokenizer = load_base_model_and_tokenizer(config)
    model = prepare_model_for_training(model, config)

    train_ds, eval_ds = build_datasets(config, tokenizer)
    if args.dry_run:
        train_ds = train_ds.select(range(min(5, len(train_ds))))
        eval_ds = eval_ds.select(range(min(2, len(eval_ds))))

    trainer = build_trainer(model, tokenizer, train_ds, eval_ds, config, dry_run=args.dry_run)
    trainer.train()

    final_dir = f"{config['training']['output_dir']}/final_adapter"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Adapter saved to {final_dir}")


if __name__ == "__main__":
    main()