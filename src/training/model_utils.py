import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_base_model_and_tokenizer(config: dict):
    model_id = config["model"]["base_model_id"]
    use_4bit = config["model"]["load_in_4bit"] and torch.cuda.is_available()

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map=config["environment"]["device_map"] if torch.cuda.is_available() else "cpu",
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=config["model"]["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_lora_config(config: dict) -> LoraConfig:
    return LoraConfig(**config["lora"])


def load_finetuned_model(base_model_id: str, adapter_path: str, trust_remote_code: bool = False):
    """Loads the base model and attaches a trained LoRA adapter — used by evaluation and inference."""
    from pathlib import Path
    from peft import PeftModel

    resolved_adapter_path = Path(adapter_path).resolve().as_posix()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(resolved_adapter_path, trust_remote_code=trust_remote_code)
    model = PeftModel.from_pretrained(base_model, resolved_adapter_path)
    return model, tokenizer
