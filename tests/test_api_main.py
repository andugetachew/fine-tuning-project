"""
Tests api/main.py's routes with model loading mocked out — verifies
routing, error handling, and response shaping without downloading a
real model.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model_and_tokenizer():
    model = MagicMock()
    model.device = "cpu"
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    tokenizer.pad_token_id = 0
    tokenizer.decode.return_value = "mocked model response"

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = {"input_ids": MagicMock(shape=[1, 5])}
    tokenizer.return_value = mock_inputs

    model.generate.return_value = [MagicMock()]
    return model, tokenizer


def test_health_and_generate_endpoints(mock_model_and_tokenizer):
    model, tokenizer = mock_model_and_tokenizer

    with patch("api.main.load_base_model_and_tokenizer", return_value=(model, tokenizer)), \
         patch("api.main.load_finetuned_model", return_value=(model, tokenizer)):
        from api.main import app

        with TestClient(app) as client:
            health_response = client.get("/health")
            assert health_response.status_code == 200
            assert health_response.json()["base_model_loaded"] is True
            assert health_response.json()["adapter_loaded"] is True

            generate_response = client.post("/generate", json={"prompt": "test prompt", "use_finetuned": True})
            assert generate_response.status_code == 200
            body = generate_response.json()
            assert body["model_used"] == "finetuned"
            assert body["prompt"] == "test prompt"

            base_response = client.post("/generate", json={"prompt": "test prompt", "use_finetuned": False})
            assert base_response.status_code == 200
            assert base_response.json()["model_used"] == "base"


def test_generate_returns_503_when_adapter_not_loaded(mock_model_and_tokenizer):
    model, tokenizer = mock_model_and_tokenizer

    with patch("api.main.load_base_model_and_tokenizer", return_value=(model, tokenizer)), \
         patch("api.main.load_finetuned_model", side_effect=Exception("adapter not found")):
        from api.main import app

        with TestClient(app) as client:
            response = client.post("/generate", json={"prompt": "test", "use_finetuned": True})
            assert response.status_code == 503