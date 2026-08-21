# CPU-only inference image — matches the project's actual deployment reality:
# no dedicated GPU, training happens on Colab, this container only serves a
# small 1.5B model + LoRA adapter for demo/portfolio purposes (Render-style
# free/starter tier). Swap the base image to a CUDA one later only if this
# is ever deployed behind a real GPU instance.

FROM python:3.12-slim-bookworm

WORKDIR /app

# System deps needed by tokenizers/sentencepiece builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY models/checkpoints/ ./models/checkpoints/

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
