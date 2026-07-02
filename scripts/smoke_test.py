"""
SSI v2 — vLLM smoke test (Lambda Cloud).

Loads 10 prompts (AdvBench/HarmBench), runs Llama 3.1 8B Instruct via vLLM at t=0.0
with strict determinism flags, prints the responses, and writes them to
smoke_test_output.json.

This is the minimum viable pipeline for Experiment 1 (determinism isolation).
Do NOT run locally — requires a GPU box with vllm/torch installed
(`uv sync --extra cloud`).
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import time
from pathlib import Path

import pandas as pd
import torch

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_CSV = REPO_ROOT / "data" / "prompts" / "prompts.csv"
OUTPUT_JSON = REPO_ROOT / "smoke_test_output.json"

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
N_PROMPTS = 10
TEMPERATURE = 0.0
SEED = 42
MAX_TOKENS = 512


def load_prompts(n: int) -> list[dict]:
    df = pd.read_csv(PROMPTS_CSV).head(n)
    return df[["id", "prompt"]].to_dict(orient="records")


def main() -> None:
    print(f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    print(f"torch deterministic: {torch.are_deterministic_algorithms_enabled()}")

    prompts = load_prompts(N_PROMPTS)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_CSV}")

    llm = LLM(
        model=MODEL,
        dtype="auto",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
        seed=SEED,
    )

    start = time.time()
    outputs = llm.generate(formatted, sampling)
    elapsed = time.time() - start

    results = []
    for prompt, output in zip(prompts, outputs):
        response = output.outputs[0].text.strip()
        print("=" * 70)
        print(f"[{prompt['id']}] {prompt['prompt']}")
        print("-" * 70)
        print(response)
        results.append(
            {
                "prompt_id": prompt["id"],
                "prompt": prompt["prompt"],
                "response": response,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "seed": SEED,
            }
        )

    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "n_prompts": len(results),
        "elapsed_seconds": round(elapsed, 2),
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "torch_manual_seed": 0,
        },
        "results": results,
    }

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))
    print("=" * 70)
    print(f"Wrote {len(results)} results to {OUTPUT_JSON} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
