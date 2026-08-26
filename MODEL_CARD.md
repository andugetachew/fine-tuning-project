# Model Card: Backend Engineering Assistant

## Base Model
- **Base model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Adapter type**: LoRA (r=16, alpha=32, target_modules=q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

## Training Data
- **Examples**: 24 real, verified backend engineering incidents
- **Domain**: Django/DRF, PostgreSQL, Docker, async systems, Python internals, ML pipelines
- **Train/eval split**: 21 train / 3 eval

## Training
- **Epochs**: 3
- **Batch size**: 2 (effective: 16 with gradient accumulation)
- **Learning rate**: 0.0002

## Evaluation
- **Scorer**: exact_match
- **Result**: see `outputs/evaluation/aggregate_stats.json` for mean scores,
  confidence intervals, and significance testing.

## Known Limitations
- Small training set (24 examples) — not expected to produce a
  large, statistically robust capability shift.
- Eval set is small (3 examples) — confidence intervals are wide;
  treat results as directional, not conclusive.
- `llm_judge` and `ai_eval_framework` scorers depend on external services
  and won't run without credentials/connectivity.

## Intended Use
A portfolio/demonstration project showing an end-to-end QLoRA fine-tuning
pipeline: data curation, training, evaluation, and serving. Not intended
for production use without a substantially larger, more rigorously
validated dataset.
