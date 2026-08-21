# Backend Engineering Assistant — QLoRA Fine-Tuning Project

![Tests](https://img.shields.io/badge/tests-18%20passed-brightgreen) ![Python](https://img.shields.io/badge/python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-enabled-009688) ![PyTorch](https://img.shields.io/badge/PyTorch-enabled-EE4C2C) ![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange) ![Docker](https://img.shields.io/badge/Docker-enabled-blue) ![License](https://img.shields.io/badge/license-MIT-yellow)

A QLoRA fine-tune of Qwen2.5-1.5B-Instruct, trained on 24 real, verified
backend engineering incidents from my own projects — Django/DRF, PostgreSQL
(sync/async drivers, migrations), Docker, async systems (Celery/asyncio),
Python internals (thread safety, serialization), and ML pipeline engineering.
Built end-to-end: data curation, training, evaluation, and serving.

## Pipeline

Raw Q&A (data/raw/)
-> Data prep (src/data/prepare_dataset.py) -> ChatML train/eval splits
-> QLoRA fine-tuning (src/train.py, Colab GPU) -> LoRA adapter
-> Evaluation (src/evaluation/evaluate.py) -> base vs fine-tuned comparison
-> Inference API (api/main.py) -> Docker


## What this demonstrates

- **Data curation discipline**: every training example is a real bug I
  personally diagnosed and fixed, not synthetic Q&A — sourced from actual
  incident writeups across multiple production projects.
- **Config-driven training**: swapping base models or hyperparameters is a
  YAML change (`configs/qlora.yaml`), not a code change. `trl`/`transformers`
  version differences are handled defensively via `inspect.signature`
  introspection rather than hardcoded argument names.
- **Genuinely pluggable evaluation**: four working scorers behind one
  interface — `exact_match`, `normalized_match`, `semantic_similarity`
  (sentence-transformers, no external dependency), `llm_judge` (Claude API),
  and `ai_eval_framework` (delegates to a separate deployed project of mine).
  The latter two cleanly self-report as unavailable when their credentials
  aren't present, rather than crashing or silently no-op'ing.
- **Real serving**: FastAPI inference API with live base/fine-tuned
  comparison mode (`use_finetuned: true/false` on the same endpoint).
- **Tested, not just run**: 18 tests covering data transformation
  correctness (not just "doesn't crash"), scorer behavior, request
  validation, and config loading.

## Results

Trained for 3 epochs on 21 examples (3 held out for eval). Eval loss
decreased steadily across all 3 epochs (3.008 → 2.833 → 2.787), confirming
genuine learning rather than noise.

On semantic-similarity evaluation against held-out examples, results were
mixed rather than a clean win for the fine-tuned model — expected at this
scale, since the base model already handles generic backend troubleshooting
competently. Manual inspection showed the fine-tuned model's answers were
more structured and closer to the training data's actual conventions
(specific commands, consistent diagnostic ordering) without a large jump in
similarity score. This is reported honestly rather than oversold: a
meaningful, statistically significant capability shift would need a larger
training and eval set than 24/3 examples.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Prepare data

```bash
python -m src.data.prepare_dataset
```

## 2. Train

Local CPU dry run (sanity check only):
```bash
python -m src.train --dry-run
```

Real training — Colab (free T4 GPU) via `notebooks/train_colab.ipynb`, or:
```bash
python -m src.train
```

## 3. Evaluate

```bash
python -m src.evaluation.evaluate --adapter models/checkpoints/final_adapter
```
Scorer is set via `evaluation.scorer_type` in `configs/qlora.yaml`
(default: `semantic_similarity`). `llm_judge` needs `ANTHROPIC_API_KEY`;
`ai_eval_framework` needs no key by default (points at a public deployed
instance) but respects `AI_EVAL_FRAMEWORK_URL`/`AI_EVAL_FRAMEWORK_API_KEY`
if set.

## 4. Serve

```bash
uvicorn api.main:app --reload
```
`POST /generate` with `{"prompt": "...", "use_finetuned": true}`.

## 5. Test

```bash
pytest -v
```
18/18 passing.

## 6. Docker

```bash
docker build -t backend-assistant .
docker run -p 8000:8000 backend-assistant
```
Builds cleanly (CPU-only base image, PyTorch installed from the CPU wheel
index to avoid pulling unnecessary CUDA packages). The base model downloads
from Hugging Face on first container start; mount a persistent volume at
`/app/.cache/huggingface` to avoid re-downloading on every run.

## Design notes

- **Swappable base model**: nothing outside `configs/qlora.yaml` names the
  model. Changing `model.base_model_id` and `lora.target_modules` is the
  entire migration path to a different model.
- **No invented numbers**: evaluation always reports raw predictions; a
  score is only reported where a real, working scorer produced it.

---

## 📄 License

MIT License

---

---

**Andualem Getachew**
[![GitHub](https://img.shields.io/badge/GitHub-andugetachew-black?logo=github)](https://github.com/andugetachew)
[![Email](https://img.shields.io/badge/Email-andugeta41%40gmail.com-red?logo=gmail)](mailto:andugeta41@gmail.com)