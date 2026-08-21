"""
Tests api/schemas.py validation logic directly — no model loading, so
these run without any GPU/CPU model weight download in CI.
"""
import pytest
from pydantic import ValidationError

from api.schemas import GenerateRequest


def test_generate_request_defaults():
    req = GenerateRequest(prompt="How do I fix a 500 error?")
    assert req.use_finetuned is True
    assert req.temperature == 0.2
    assert req.max_new_tokens == 256


def test_generate_request_rejects_empty_prompt():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="")


def test_generate_request_rejects_out_of_range_temperature():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="test", temperature=3.0)
