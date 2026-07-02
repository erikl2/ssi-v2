# SSI v2 — How stable are safety evaluations under real inference?

> **Preview.** Experiment 4 (closed-source production APIs) is complete and reported
> below. Experiments 1–3 and 5 are in progress — see the roadmap. This page will fill
> in as each result lands.

This repository is the second empirical artifact in a line of work on the
**reproducibility of LLM safety evaluations**. SSI v1 (arXiv:2512.12066) showed that
18–28% of harmful prompts flip between *refuse* and *comply* across sampling
configurations in a controlled vLLM setting. A natural objection followed: *that
instability was an artifact of an artificial setup — real production inference would
behave differently, and probably worse.* v2 tests that objection head-on.

## The question

How much does a model's refuse-vs-comply verdict move when nothing about the prompt
changes and you only re-run it? Per prompt we measure two things:

- **SSI** — the share of prompts whose verdict flips at least once across repeated,
  identical calls (lower = more stable).
- **Mean flip rate** — the average per-prompt probability that the verdict changes.

## What Experiment 4 found (closed-source APIs, no seed control)

Three frontier models, a stratified 200-prompt sample, 5 reruns each (~3,000 calls),
with no determinism flags or seed control of any kind:

| Model | SSI (prompts that flip) | Mean flip rate |
|---|---|---|
| claude-opus-4-7 | 0.5% | 0.20% |
| claude-sonnet-4-5 | 1.1% | 0.22% |
| gpt-4o-2024-11-20 | 2.5% | 0.60% |

SSI is computed over calls that actually reached the model. Infra-layer content-filter
errors are their own category, excluded from every numerator and denominator (see
Limitations); 20 of Sonnet's 200 prompts were fully content-filtered and drop out, so its
rates are over 180 prompts.

**1. Frontier production APIs are highly stable.** Single-digit-percent or lower, versus
18–28% on the small open models in v1 — even with zero determinism controls. Only 8 of the
580 (prompt, model) pairs that returned usable responses flipped at all, and
clearly-harmful prompts showed **0% instability** across all three models: they refuse
reliably. The residual instability
lives entirely in borderline / contested prompts (politics, copyright, persuasion on
disputed claims). That is the direct empirical answer to the "production inference is
worse" objection: on these models, it isn't.

**2. The interesting instability is a value judgment, not noise.** On a prompt asking for
a persuasive article defending a contested territorial annexation as "democratic," the
three models make three *different but internally stable* choices: one complies every
time, one refuses every time, and only the third actually flips (4 of 5 complies). The
disagreement is *across* models, not *within* a model's randomness — a sharper result
than any single aggregate number. Where models are unstable, it is because the prompt is
genuinely ambiguous and they resolve it differently.

### Methods note — why we validated before believing it

An initial pass reported one model at ~19.5% instability. A spot-check flagged it as an
artifact: that model phrases refusals with a curly apostrophe (U+2019) while the
refusal-detection patterns used straight apostrophes, so identical refusals were counted
as flips. Normalizing smart punctuation dropped it to its true 2.5% — an 8× false
headline avoided. The classifier is deliberately conservative; see Limitations.

## Limitations

- **Binary refuse/comply.** Soft redirects ("I can help you think this through, but I
  won't write that") are currently scored as *comply*, so true SSI may be marginally
  lower than reported. An LLM-judge scoring pass would tighten this; it is planned for the
  full paper, not this preview.
- Experiment 4 uses a stratified 200-prompt sample of the v1 prompt set, not all 876.
- **Content-filter errors are excluded, not counted as refusals.** 105 of Sonnet's 1,000
  calls (10.5%) were blocked by the provider's content filter before reaching the model —
  20 prompts were blocked on all 5 calls. These are an infra-layer event, not a model
  refusal decision, so they are their own category and excluded from every rate's
  numerator and denominator. Opus and GPT-4o had zero such errors. (An earlier version of
  the analysis folded these into "refuse," which inflated Sonnet's SSI to 2.0%.)

## Layout

```
ssi_v2/
├── data/
│   ├── prompts/
│   │   ├── prompts.csv        # 876 prompts (491 AdvBench + 385 HarmBench, copied from v1)
│   │   └── exp4_sample.csv    # stratified 200-prompt sample for Exp 4
│   └── results/exp4/          # Exp 4 output (calls.jsonl, analysis; gitignored)
├── scripts/
│   ├── smoke_test.py          # vLLM smoke (Lambda Cloud)
│   ├── api_smoke.py           # OpenAI + Anthropic smoke (laptop-runnable)
│   ├── run_exp4.py            # Exp 4 driver (resumable; ~3,000 calls, ~$20–25)
│   └── analyze_exp4.py        # Exp 4 stability analysis → SSI / flip rates
└── pyproject.toml             # deps; install [cloud] extra on GPU box only
```

## Setup & reproduction

```bash
# Laptop (API track — Experiment 4)
uv sync
uv run python scripts/api_smoke.py
uv run python scripts/run_exp4.py --dry-run    # sanity-check the plan
uv run python scripts/run_exp4.py              # resumable; ~3,000 calls, ~$20–25
uv run python scripts/analyze_exp4.py          # SSI / flip-rate table

# Lambda Cloud (GPU track — Experiments 1–3)
uv sync --extra cloud
uv run python scripts/smoke_test.py
```

Models for Experiment 4 are pinned in `scripts/run_exp4.py`; copy `.env.example` to
`.env` and fill in your API keys first.

> **Note on torch pin.** The `[cloud]` extra pins `torch==2.4.0` to match
> `vllm==0.6.3`'s actual requirement. SSI v1's `requirements.txt` pinned
> `torch==2.4.1`, but pip's loose resolver silently downgraded to 2.4.0 at
> install time; this repo makes the real installed version explicit so
> `uv sync --extra cloud` produces a reproducible environment — important
> because Experiment 1 is specifically measuring inference determinism.

## Roadmap

| # | Experiment | Status |
|---|---|---|
| 1 | Determinism isolation (v1 rerun with determinism flags) | Planned |
| 2 | Scale (Llama 3.1 70B, Qwen 2.5 72B) | Planned |
| 3 | Production inference (vLLM spec-decoding, batch variation) | Planned |
| 4 | **Closed-source APIs** | ✅ Complete |
| 5 | Safety-training comparison (base vs DPO/RLHF) | Stretch |

Target venue: a NeurIPS 2026 workshop, with an Alignment Forum writeup.

## Prior work

SSI v1 — arXiv:2512.12066.
