# SSI v2 — Safety Eval Reliability Under Production Inference

SSI v1 (arXiv:2512.12066) showed that 18–28% of harmful prompts flip between
refuse/comply across sampling configurations in a controlled vLLM setting.
v2 asks how much worse this problem is under realistic conditions: with strict
determinism flags enabled (Exp 1), at 70B+ scale (Exp 2), under production
inference features like speculative decoding and variable batch sizes (Exp 3),
through seedless closed-source APIs (Exp 4), and across different safety
post-training regimes (Exp 5). The goal is a second empirical artifact that
directly addresses the "production inference introduces more non-determinism"
critique raised against v1, with a target of a NeurIPS 2026 submission and an
Alignment Forum writeup.

## Experiments

1. **Determinism isolation** — rerun v1 with `CUBLAS_WORKSPACE_CONFIG=:4096:8`
   and `torch.use_deterministic_algorithms(True)`; compare flip rates.
2. **Scale testing** — Llama 3.1 70B + Qwen 2.5 72B on the same 876 prompts.
3. **Production inference** — vLLM with spec decoding on/off, varied batch sizes.
4. **Closed-source APIs** — gpt-4o-2024-11-20, claude-sonnet-4-5-20250929,
   claude-opus-4-7 (pinned in `scripts/run_exp4.py`), no seed control.
5. **(Stretch) Safety training comparison** — base instruct vs. DPO/RLHF variants.

## Layout

```
ssi_v2/
├── data/
│   ├── prompts/
│   │   ├── prompts.csv        # 876 BeaverTails prompts (copied from v1)
│   │   └── exp4_sample.csv    # stratified 200-prompt sample for Exp 4
│   └── results/exp4/          # Exp 4 output (calls.jsonl; gitignored)
├── scripts/
│   ├── smoke_test.py          # vLLM smoke (Lambda Cloud)
│   ├── api_smoke.py           # OpenAI + Anthropic smoke (laptop-runnable)
│   └── run_exp4.py            # Exp 4 driver (resumable; ~3,000 calls, ~$20–25)
└── pyproject.toml             # deps; install [cloud] extra on GPU box only
```

## Setup

```bash
# Laptop (API track — Exp 4)
uv sync
uv run python scripts/api_smoke.py

# Exp 4 (resumable; rerun the same command after a crash)
uv run python scripts/run_exp4.py --dry-run   # sanity-check the plan first
uv run python scripts/run_exp4.py

# Lambda Cloud (Exps 1–3)
uv sync --extra cloud
uv run python scripts/smoke_test.py
```

> **Note on torch pin.** The `[cloud]` extra pins `torch==2.4.0` to match
> `vllm==0.6.3`'s actual requirement. SSI v1's `requirements.txt` pinned
> `torch==2.4.1`, but pip's loose resolver silently downgraded to 2.4.0 at
> install time; this repo makes the real installed version explicit so
> `uv sync --extra cloud` produces a reproducible environment — important
> because Experiment 1 is specifically measuring inference determinism.
