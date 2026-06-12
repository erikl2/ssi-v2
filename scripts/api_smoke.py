"""
SSI v2 — API smoke test (Experiment 4, laptop-runnable).

Loads 10 BeaverTails prompts, hits gpt-4o-2024-11-20 and
claude-haiku-4-5-20251001 2x each per prompt (no seed control; the point is to
measure what real deployers actually get), and writes timestamped responses to
api_smoke_output.json.

Requires env vars OPENAI_API_KEY and ANTHROPIC_API_KEY — read from the process
environment or a .env file at the repo root.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_CSV = REPO_ROOT / "data" / "prompts" / "prompts.csv"
OUTPUT_JSON = REPO_ROOT / "api_smoke_output.json"

load_dotenv(REPO_ROOT / ".env")

from anthropic import Anthropic
from openai import OpenAI

N_PROMPTS = 10
N_CALLS_PER_MODEL = 2
MAX_TOKENS = 512

OPENAI_MODEL = "gpt-4o-2024-11-20"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def load_prompts(n: int) -> list[dict]:
    df = pd.read_csv(PROMPTS_CSV).head(n)
    return df[["id", "prompt"]].to_dict(orient="records")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def call_openai(client: OpenAI, prompt: str) -> dict:
    start = time.time()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )
    return {
        "provider": "openai",
        "model": OPENAI_MODEL,
        "response": resp.choices[0].message.content or "",
        "finish_reason": resp.choices[0].finish_reason,
        "timestamp": now_iso(),
        "latency_seconds": round(time.time() - start, 3),
    }


def call_anthropic(client: Anthropic, prompt: str) -> dict:
    start = time.time()
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return {
        "provider": "anthropic",
        "model": ANTHROPIC_MODEL,
        "response": text,
        "finish_reason": resp.stop_reason,
        "timestamp": now_iso(),
        "latency_seconds": round(time.time() - start, 3),
    }


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    openai_client = OpenAI()
    anthropic_client = Anthropic()

    prompts = load_prompts(N_PROMPTS)
    print(f"Loaded {len(prompts)} prompts; {N_CALLS_PER_MODEL} calls per model per prompt")

    results = []
    for i, p in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {p['id']} — {p['prompt'][:70]}")
        calls = []
        for call_idx in range(N_CALLS_PER_MODEL):
            calls.append({"call_index": call_idx, **call_openai(openai_client, p["prompt"])})
            calls.append({"call_index": call_idx, **call_anthropic(anthropic_client, p["prompt"])})
        results.append({"prompt_id": p["id"], "prompt": p["prompt"], "calls": calls})

    payload = {
        "openai_model": OPENAI_MODEL,
        "anthropic_model": ANTHROPIC_MODEL,
        "n_prompts": len(results),
        "calls_per_model_per_prompt": N_CALLS_PER_MODEL,
        "started_at": results and results[0]["calls"][0]["timestamp"],
        "finished_at": now_iso(),
        "results": results,
    }

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))
    total_calls = len(results) * N_CALLS_PER_MODEL * 2
    print(f"Wrote {total_calls} calls for {len(results)} prompts to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
