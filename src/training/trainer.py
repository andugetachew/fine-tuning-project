from pathlib import Path

from datasets import load_dataset
from peft import get_peft_model
from trl import SFTTrainer, SFTConfig

from src.training.model_utils import build_lora_config


def format_chatml(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def build_datasets(config: dict, tokenizer):
    ds = config["dataset"]
    train_ds = load_dataset("json", data_files=ds["train_path"], split="train")
    eval_ds = load_dataset("json", data_files=ds["eval_path"], split="train")

    train_ds = train_ds.map(lambda ex: format_chatml(ex, tokenizer), remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(lambda ex: format_chatml(ex, tokenizer), remove_columns=eval_ds.column_names)
    return train_ds, eval_ds

def build_trainer(model, tokenizer, train_ds, eval_ds, config: dict, dry_run: bool = False):
    import inspect
    import torch

    tcfg = config["training"]
    output_dir = Path(tcfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available()

    desired_args = {
        "output_dir": str(output_dir),
        "num_train_epochs": 1 if dry_run else tcfg["num_train_epochs"],
        "per_device_train_batch_size": tcfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": tcfg["gradient_accumulation_steps"],
        "learning_rate": tcfg["learning_rate"],
        "lr_scheduler_type": tcfg["lr_scheduler_type"],
        "warmup_ratio": tcfg["warmup_ratio"],
        "logging_steps": tcfg["logging_steps"],
        "save_strategy": tcfg["save_strategy"],
        "eval_strategy": tcfg["eval_strategy"],
        "bf16": tcfg["bf16"] and has_cuda,
        "use_cpu": not has_cuda,
        "optim": "adamw_torch" if not has_cuda else tcfg["optim"],
        "report_to": tcfg["report_to"],
        "dataset_text_field": "text",
        "max_seq_length": config["tokenizer"]["max_seq_length"],
        "packing": False,
        "dataloader_num_workers": 0,
    }

    accepted_config_params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    filtered_args = {k: v for k, v in desired_args.items() if k in accepted_config_params}
    skipped = set(desired_args) - set(filtered_args)
    if skipped:
        print(f"Note: this trl version's SFTConfig doesn't accept {skipped} — skipped, using its defaults instead.")

    sft_config = SFTConfig(**filtered_args)

    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }

    accepted_trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    if "tokenizer" in accepted_trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in accepted_trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    # else: newer trl infers the tokenizer from the model automatically, nothing to pass

    return SFTTrainer(**trainer_kwargs)


def prepare_model_for_training(model, config: dict):
    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
