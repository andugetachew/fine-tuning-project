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
def test_load_config_returns_dict_with_expected_structure(tmp_path):
    import yaml
    from src.training.model_utils import load_config

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(
        yaml.dump({"model": {"base_model_id": "test/model"}, "lora": {"r": 8, "task_type": "CAUSAL_LM"}}),
        encoding="utf-8",
    )
    config = load_config(str(config_path))
    assert config["model"]["base_model_id"] == "test/model"
    assert config["lora"]["r"] == 8