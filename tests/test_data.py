"""
Validates the dataset schema and the ChatML conversion — catches broken
raw JSONL before it ever reaches training, which is cheaper to fix here
than after burning Colab GPU time.
"""
import json
from pathlib import Path

import pytest

from src.constants import SYSTEM_PROMPT
from src.data.prepare_dataset import to_chatml, load_raw_examples


def test_raw_examples_have_required_fields():
    for ex in load_raw_examples():
        assert "instruction" in ex
        assert "output" in ex
        assert isinstance(ex["instruction"], str) and ex["instruction"].strip()
        assert isinstance(ex["output"], str) and ex["output"].strip()


def test_to_chatml_produces_three_turn_conversation():
    example = {"instruction": "Why does X fail?", "input": "", "output": "Because Y."}
    result = to_chatml(example)
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert result["messages"][-1]["content"] == "Because Y."


def test_to_chatml_uses_shared_system_prompt():
    example = {"instruction": "Q", "input": "", "output": "A"}
    result = to_chatml(example)
    assert result["messages"][0]["content"] == SYSTEM_PROMPT


def test_to_chatml_preserves_instruction_content_exactly():
    example = {"instruction": "What does the DB_HOST env var control?", "input": "", "output": "A"}
    result = to_chatml(example)
    assert result["messages"][1]["content"] == "What does the DB_HOST env var control?"


def test_to_chatml_appends_input_when_present():
    example = {"instruction": "Explain this error", "input": "TypeError: bad argument", "output": "A"}
    result = to_chatml(example)
    user_content = result["messages"][1]["content"]
    assert "Explain this error" in user_content
    assert "TypeError: bad argument" in user_content


def test_to_chatml_does_not_add_extra_content_when_input_empty():
    example = {"instruction": "Explain this error", "input": "", "output": "A"}
    result = to_chatml(example)
    assert result["messages"][1]["content"] == "Explain this error"


def test_processed_splits_are_valid_jsonl_if_present():
    for split_path in [Path("data/processed/train.jsonl"), Path("data/processed/eval.jsonl")]:
        if not split_path.exists():
            pytest.skip(f"{split_path} not generated yet — run prepare_dataset.py first")
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                assert "messages" in record
                assert len(record["messages"]) == 3


def test_write_jsonl_and_main_produce_valid_splits(tmp_path, monkeypatch):
    import src.data.prepare_dataset as pd

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()

    raw_file = raw_dir / "sample.jsonl"
    raw_file.write_text(
        '{"instruction": "Q1", "input": "", "output": "A1"}\n'
        '{"instruction": "Q2", "input": "", "output": "A2"}\n'
        '{"instruction": "Q3", "input": "", "output": "A3"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(pd, "RAW_DIR", raw_dir)
    monkeypatch.setattr(pd, "PROCESSED_DIR", processed_dir)

    pd.main()

    train_path = processed_dir / "train.jsonl"
    eval_path = processed_dir / "eval.jsonl"
    assert train_path.exists()
    assert eval_path.exists()

    train_lines = train_path.read_text(encoding="utf-8").strip().splitlines()
    eval_lines = eval_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) + len(eval_lines) == 3
    for line in train_lines + eval_lines:
        record = json.loads(line)
        assert len(record["messages"]) == 3


def test_main_raises_systemexit_when_no_raw_examples(tmp_path, monkeypatch):
    import src.data.prepare_dataset as pd

    monkeypatch.setattr(pd, "RAW_DIR", tmp_path / "empty_raw")
    (tmp_path / "empty_raw").mkdir()
    monkeypatch.setattr(pd, "PROCESSED_DIR", tmp_path / "processed")

    with pytest.raises(SystemExit):
        pd.main()