"""
Fast, CPU-only tests for the training pipeline's plumbing — not full
training runs. These should complete in seconds and catch config/wiring
bugs before a Colab run.
"""
from src.training.model_utils import load_config, build_lora_config


def test_config_loads_expected_sections():
    config = load_config("configs/qlora.yaml")
    for section in ["model", "lora", "tokenizer", "dataset", "training", "environment", "evaluation"]:
        assert section in config, f"Missing '{section}' section in qlora.yaml"


def test_lora_config_builds_from_yaml():
    config = load_config("configs/qlora.yaml")
    lora_config = build_lora_config(config)
    assert lora_config.r == config["lora"]["r"]
    assert lora_config.task_type == config["lora"]["task_type"]
