# Backend Engineering Assistant — QLoRA Fine-Tuning Project

![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen) ![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen) ![Python](https://img.shields.io/badge/python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-enabled-009688) ![PyTorch](https://img.shields.io/badge/PyTorch-enabled-EE4C2C) ![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange) ![Docker](https://img.shields.io/badge/Docker-enabled-blue) ![License](https://img.shields.io/badge/license-MIT-yellow)

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
-> Aggregate stats (src/evaluation/stats.py) -> bootstrap CI + significance test
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
- **Statistical rigor**: bootstrap confidence intervals and a paired
  Wilcoxon significance test on base-vs-fine-tuned comparisons, plus a
  multi-run comparison tool for comparing different training runs against
  each other — not just raw score averages.
- **Real serving**: FastAPI inference API with live base/fine-tuned
  comparison mode (`use_finetuned: true/false` on the same endpoint), a
  streaming (`/generate/stream`) endpoint, and request logging that builds
  a real corpus of candidate future training examples.
- **Production-style deployment path**: LoRA adapter merging
  (`merge_and_unload()`) for single-artifact inference, alongside the
  default swappable-adapter path.
- **Auto-generated model card**: documents base model, LoRA config, and
  known limitations directly from the actual training config — not
  hand-written and prone to drifting out of sync.
- **Tested, not just run**: 38 tests, 86% coverage on unit-testable code —
  data transformation correctness, scorer behavior (including real calls
  to a live deployed evaluation service), statistics computation, API
  routing with mocked models, and CLI tools. See "Test coverage" below for
  what's intentionally excluded and why.
- **CI**: GitHub Actions runs the full test suite on every push.

## Results

Trained for 3 epochs on 21 examples (3 held out for eval). Eval loss
decreased steadily across all 3 epochs (3.008 → 2.833 → 2.787), confirming
genuine learning rather than noise.

On semantic-similarity evaluation against 3 held-out examples:

| | Mean | 95% CI |
|---|---|---|
| Base model | 0.613 | [0.520, 0.704] |
| Fine-tuned model | 0.627 | [0.531, 0.693] |

A paired Wilcoxon signed-rank test found this difference **not statistically
significant** (p = 0.75) — expected and honest at this sample size (n=3).
The confidence intervals overlap substantially, so no claim of measurable
improvement is being made here. Manual inspection of individual responses
showed the fine-tuned model's answers were more structured and closer to
the training data's conventions (specific commands, consistent diagnostic
ordering), but this qualitative observation doesn't yet have statistical
backing. A larger training and eval set (50-100+ examples each) would be
the natural next step to detect a real effect if one exists.

Full statistical methodology (bootstrap confidence intervals, Wilcoxon
significance testing) is in `src/evaluation/stats.py`. See `MODEL_CARD.md`
for full training/architecture details.

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

### Aggregate statistics & model card

```bash
python -m src.evaluation.stats
python -m src.evaluation.model_card
```
Produces `outputs/evaluation/aggregate_stats.json` (bootstrap CIs + Wilcoxon
significance test) and `MODEL_CARD.md` (auto-generated from the actual
training config).

### Comparing multiple runs

```bash
python -m src.evaluation.compare_runs --runs r16=outputs/eval_r16.json r32=outputs/eval_r32.json
```
Same statistical rigor as base-vs-fine-tuned, generalized to compare any
number of named evaluation runs (e.g. different LoRA ranks or training
configs) against each other.

## 4. Serve

```bash
uvicorn api.main:app --reload
```
- `POST /generate` with `{"prompt": "...", "use_finetuned": true}` — full response.
- `POST /generate/stream` — same request shape, streams tokens as they generate.
- Every request/response is logged to `outputs/predictions/request_log.jsonl`
  as a growing pool of candidate future training examples.

## 5. Test

```bash
pytest --cov=src --cov=api --cov-report=term-missing
```
38/38 passing, 86% coverage.

### Test coverage

`.coveragerc` excludes `src/train.py`, `src/training/trainer.py`,
`src/training/merge_adapter.py`, and `src/evaluation/evaluate.py` from
coverage reporting. These require downloading and running a real ~3GB
model to exercise meaningfully — unit-testing them would mean either
downloading the model in CI (slow, costly) or mocking so heavily the test
verifies nothing real. They're instead validated by actually running the
training/evaluation pipeline end to end (see Results above), the same
approach used for the equivalent training scripts in my `ai-eval-framework`
project.

## 6. Merge adapter (optional, production-style deployment)

```bash
python -m src.training.merge_adapter
```
Bakes the LoRA weights into the base model, producing a single standalone
model directory (`models/merged/`) — faster inference, simpler deployment,
at the cost of losing the ability to swap adapters without reloading the
full model.

## 7. Docker

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

**Andualem Getachew**
[![GitHub](https://img.shields.io/badge/GitHub-andugetachew-black?logo=github)](https://github.com/andugetachew)
[![Email](https://img.shields.io/badge/Email-andugeta41%40gmail.com-red?logo=gmail)](mailto:andugeta41@gmail.com)