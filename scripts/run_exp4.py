"""
SSI v2 — Experiment 4: closed-source API safety stability.

Runs the stratified 200-prompt sample (data/prompts/exp4_sample.csv) through
three frontier APIs, 5× per prompt per model, writing one JSONL record per
call. Supports resume-from-crash and per-model filtering.

Models (all versioned):
  - gpt-4o-2024-11-20
  - claude-sonnet-4-5-20250929
  - claude-opus-4-7

Total calls at full tilt: 200 × 5 × 3 = 3,000. Estimated cost ~$20–25.

Usage:
    # Full run (all 3 models)
    uv run python scripts/run_exp4.py

    # Single model (cheap sanity pass)
    uv run python scripts/run_exp4.py --models gpt-4o-2024-11-20

    # Cap at 10 calls total, no API calls (dry run)
    uv run python scripts/run_exp4.py --dry-run --limit 10

    # Resume after crash — already-recorded (prompt_id, model, call_idx)
    # triples are skipped automatically
    uv run python scripts/run_exp4.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO_ROOT / "data" / "prompts" / "exp4_sample.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "exp4"
OUTPUT_JSONL = OUTPUT_DIR / "calls.jsonl"

load_dotenv(REPO_ROOT / ".env")

from anthropic import Anthropic, APIError as AnthropicAPIError
from openai import OpenAI, APIError as OpenAIAPIError

OPENAI_MODELS = {"gpt-4o-2024-11-20"}
ANTHROPIC_MODELS = {"claude-sonnet-4-5-20250929", "claude-opus-4-7"}
DEFAULT_MODELS = list(OPENAI_MODELS | ANTHROPIC_MODELS)

CALLS_PER_PROMPT_PER_MODEL = 5
MAX_TOKENS = 512


@dataclass
class CallPlan:
    prompt_id: str
    stratum: str
    prompt: str
    model: str
    call_idx: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_sample() -> pd.DataFrame:
    if not SAMPLE_CSV.exists():
        raise SystemExit(f"Missing sample CSV: {SAMPLE_CSV}")
    df = pd.read_csv(SAMPLE_CSV)
    required = {"prompt_id", "stratum", "prompt"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Sample CSV missing columns: {missing}")
    return df


def build_plan(df: pd.DataFrame, models: list[str]) -> list[CallPlan]:
    plan: list[CallPlan] = []
    for _, row in df.iterrows():
        for model in models:
            for call_idx in range(CALLS_PER_PROMPT_PER_MODEL):
                plan.append(
                    CallPlan(
                        prompt_id=row["prompt_id"],
                        stratum=row["stratum"],
                        prompt=row["prompt"],
                        model=model,
                        call_idx=call_idx,
                    )
                )
    return plan


def load_completed(jsonl_path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "ok":
                continue
            done.add((rec["prompt_id"], rec["model"], rec["call_idx"]))
    return done


def call_openai(client: OpenAI, model: str, prompt: str) -> dict:
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )
    usage = resp.usage
    return {
        "provider": "openai",
        "response": resp.choices[0].message.content or "",
        "finish_reason": resp.choices[0].finish_reason,
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "latency_seconds": round(time.time() - start, 3),
    }


def call_anthropic(client: Anthropic, model: str, prompt: str) -> dict:
    start = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return {
        "provider": "anthropic",
        "response": text,
        "finish_reason": resp.stop_reason,
        "input_tokens": getattr(resp.usage, "input_tokens", None),
        "output_tokens": getattr(resp.usage, "output_tokens", None),
        "latency_seconds": round(time.time() - start, 3),
    }


def execute_call(
    item: CallPlan,
    openai_client: OpenAI,
    anthropic_client: Anthropic,
) -> dict:
    base = {
        "prompt_id": item.prompt_id,
        "stratum": item.stratum,
        "prompt": item.prompt,
        "model": item.model,
        "call_idx": item.call_idx,
        "timestamp": now_iso(),
    }
    try:
        if item.model in OPENAI_MODELS:
            result = call_openai(openai_client, item.model, item.prompt)
        elif item.model in ANTHROPIC_MODELS:
            result = call_anthropic(anthropic_client, item.model, item.prompt)
        else:
            return {**base, "status": "error", "error": f"Unknown model: {item.model}"}
        return {**base, "status": "ok", **result}
    except (OpenAIAPIError, AnthropicAPIError) as e:
        return {**base, "status": "error", "error": f"{type(e).__name__}: {e}"}
    except Exception as e:
        return {**base, "status": "error", "error": f"{type(e).__name__}: {e}"}


def validate_models(models: list[str]) -> None:
    known = OPENAI_MODELS | ANTHROPIC_MODELS
    unknown = [m for m in models if m not in known]
    if unknown:
        raise SystemExit(
            f"Unknown model(s): {unknown}. Known: {sorted(known)}. "
            "Add to OPENAI_MODELS/ANTHROPIC_MODELS if intentional."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Model IDs to run. Default: all three. Known: {sorted(OPENAI_MODELS | ANTHROPIC_MODELS)}",
    )
    p.add_argument("--dry-run", action="store_true", help="Print plan, don't call APIs")
    p.add_argument("--limit", type=int, default=None, help="Cap total calls (for testing)")
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_JSONL,
        help=f"JSONL output path (default: {OUTPUT_JSONL})",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    validate_models(args.models)

    df = load_sample()
    plan = build_plan(df, args.models)
    output_path: Path = args.output

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)
    remaining = [c for c in plan if (c.prompt_id, c.model, c.call_idx) not in completed]
    if args.limit is not None:
        remaining = remaining[: args.limit]

    print(f"Sample:    {len(df)} prompts ({dict(df['stratum'].value_counts())})")
    print(f"Models:    {args.models}")
    print(f"Planned:   {len(plan)} calls ({len(plan) // len(args.models)}/model)")
    print(f"Completed: {len(completed)} (from {output_path})")
    print(f"Remaining: {len(remaining)}")
    if args.limit is not None:
        print(f"Limit:     capped at {args.limit}")

    if args.dry_run:
        print("[DRY RUN] Exiting before any API calls.")
        return 0

    if not remaining:
        print("Nothing to do. Exiting.")
        return 0

    import os
    if any(m in OPENAI_MODELS for m in args.models) and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (needed for requested models)")
    if any(m in ANTHROPIC_MODELS for m in args.models) and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (needed for requested models)")

    openai_client = OpenAI()
    anthropic_client = Anthropic()

    errors = 0
    with output_path.open("a") as sink:
        for item in tqdm(remaining, desc="calls"):
            rec = execute_call(item, openai_client, anthropic_client)
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            if rec.get("status") != "ok":
                errors += 1
                tqdm.write(f"  ERROR {item.model} {item.prompt_id}#{item.call_idx}: {rec.get('error')}")

    print(f"Done. Wrote {len(remaining)} records to {output_path}. Errors: {errors}.")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
