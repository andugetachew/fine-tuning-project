"""
Converts raw instruction/response JSONL into Qwen2.5 ChatML-formatted
train/eval splits ready for SFTTrainer.

Raw format expected in data/raw/*.jsonl:
    {"instruction": "...", "input": "", "output": "...", "category": "..."}

Usage:
    python -m src.data.prepare_dataset
"""
import json
import random
from pathlib import Path

from src.constants import SYSTEM_PROMPT

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
EVAL_SPLIT_RATIO = 0.1
SEED = 42


def to_chatml(example: dict) -> dict:
    user_content = example["instruction"]
    if example.get("input"):
        user_content += f"\n\n{example['input']}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def load_raw_examples() -> list[dict]:
    examples = []
    for file in RAW_DIR.glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
    return examples


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)
    raw = load_raw_examples()

    if not raw:
        raise SystemExit(
            f"No raw examples found in {RAW_DIR}/. "
            "Add at least one *.jsonl file with instruction/input/output records."
        )

    formatted = [to_chatml(ex) for ex in raw]
    random.shuffle(formatted)

    split_idx = max(1, int(len(formatted) * (1 - EVAL_SPLIT_RATIO)))
    train_examples = formatted[:split_idx]
    eval_examples = formatted[split_idx:] or formatted[-1:]

    write_jsonl(PROCESSED_DIR / "train.jsonl", train_examples)
    write_jsonl(PROCESSED_DIR / "eval.jsonl", eval_examples)

    print(f"Total raw examples : {len(raw)}")
    print(f"Train examples      : {len(train_examples)}")
    print(f"Eval examples       : {len(eval_examples)}")
    print(f"Written to          : {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
