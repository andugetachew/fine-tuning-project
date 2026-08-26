"""
Merges a trained LoRA adapter into the base model, producing a single
standalone model — no PEFT/adapter loading needed at inference time.
Faster inference, simpler deployment, at the cost of losing the ability
to swap adapters without reloading the full model.
"""
import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(base_model_id: str, adapter_path: str, output_dir: str) -> None:
    adapter_path = Path(adapter_path).resolve().as_posix()
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="models/checkpoints/final_adapter")
    parser.add_argument("--output", default="models/merged")
    args = parser.parse_args()
    merge_adapter(args.base_model, args.adapter, args.output)


if __name__ == "__main__":
    main()