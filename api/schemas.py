"""
Request/response models for the inference API.
"""
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="User instruction/question.")
    system_prompt: str | None = Field(
        default=None,
        description="Overrides the default system prompt. Leave unset to use the fine-tuning system prompt.",
    )
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    use_finetuned: bool = Field(
        default=True,
        description="If false, routes to the base model instead of the fine-tuned adapter — useful for live comparison demos.",
    )


class GenerateResponse(BaseModel):
    response: str
    model_used: str  # "base" | "finetuned"
    prompt: str


class HealthResponse(BaseModel):
    status: str
    base_model_loaded: bool
    adapter_loaded: bool
    device: str
