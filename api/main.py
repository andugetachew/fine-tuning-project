"""
FastAPI inference service for the fine-tuned backend-engineering assistant.

Loads the base model once and the fine-tuned adapter once at startup,
keeps both resident in memory, and serves either on request via
GenerateRequest.use_finetuned — this lets the API double as a live
base-vs-finetuned comparison demo, not just a single-model endpoint.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException

from api.schemas import GenerateRequest, GenerateResponse, HealthResponse
from src.constants import SYSTEM_PROMPT
from src.training.model_utils import load_config, load_base_model_and_tokenizer, load_finetuned_model

MODELS: dict = {"base": None, "base_tokenizer": None, "finetuned": None, "finetuned_tokenizer": None}
CONFIG_PATH = "configs/qlora.yaml"
ADAPTER_PATH = "models/checkpoints/final_adapter"


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config(CONFIG_PATH)
    MODELS["config"] = config

    base_model, base_tokenizer = load_base_model_and_tokenizer(config)
    MODELS["base"] = base_model
    MODELS["base_tokenizer"] = base_tokenizer

    try:
        ft_model, ft_tokenizer = load_finetuned_model(
            config["model"]["base_model_id"], ADAPTER_PATH, config["model"]["trust_remote_code"]
        )
        MODELS["finetuned"] = ft_model
        MODELS["finetuned_tokenizer"] = ft_tokenizer
    except Exception as e:
        # API still serves base-model-only if the adapter isn't trained/present yet
        # (e.g. during local development before a Colab training run has completed).
        print(f"Warning: fine-tuned adapter not loaded ({e}). Falling back to base-only mode.")
        MODELS["finetuned"] = None
        MODELS["finetuned_tokenizer"] = None

    yield
    MODELS.clear()


app = FastAPI(
    title="Backend Engineering Assistant API",
    description="Serves a QLoRA-fine-tuned instruct model specialized in practical Python/Django backend problem-solving.",
    version="0.1.0",
    lifespan=lifespan,
)


def _generate(model, tokenizer, prompt: str, system_prompt: str, max_new_tokens: int, temperature: float) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        base_model_loaded=MODELS.get("base") is not None,
        adapter_loaded=MODELS.get("finetuned") is not None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    system_prompt = request.system_prompt or SYSTEM_PROMPT

    if request.use_finetuned:
        if MODELS.get("finetuned") is None:
            raise HTTPException(
                status_code=503,
                detail="Fine-tuned adapter is not loaded yet. Train it (src/train.py) or set use_finetuned=false to use the base model.",
            )
        model, tokenizer, label = MODELS["finetuned"], MODELS["finetuned_tokenizer"], "finetuned"
    else:
        model, tokenizer, label = MODELS["base"], MODELS["base_tokenizer"], "base"

    response_text = _generate(
        model, tokenizer, request.prompt, system_prompt, request.max_new_tokens, request.temperature
    )
    return GenerateResponse(response=response_text, model_used=label, prompt=request.prompt)
