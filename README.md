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
4. **Closed-source APIs** — GPT-4o + Claude 3.5 Sonnet, no seed control.
5. **(Stretch) Safety training comparison** — base instruct vs. DPO/RLHF variants.

## Layout

```
ssi_v2/
├── data/prompts/prompts.csv   # 876 BeaverTails prompts (copied from v1)
├── scripts/
│   ├── smoke_test.py          # vLLM smoke (Lambda Cloud)
│   └── api_smoke.py           # OpenAI + Anthropic smoke (laptop-runnable)
└── pyproject.toml             # deps; install [cloud] extra on GPU box only
```

## Setup

```bash
# Laptop (API track — Exp 4)
uv sync
python scripts/api_smoke.py

# Lambda Cloud (Exps 1–3)
uv sync --extra cloud
python scripts/smoke_test.py
```
