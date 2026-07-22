#!/usr/bin/env python3
"""Compare end-to-end reward convergence across verl logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


VERL_AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERL_AGENT_ROOT))

from metrics import compute_threshold_stats, parse_step_records


def summarize(path: str, thresholds: list[float], window: int, max_step: int | None) -> dict:
    records = parse_step_records(path)
    if max_step is not None:
        records = {step: row for step, row in records.items() if step <= max_step}
    rewards = [row["critic/rewards/mean"] for row in records.values() if "critic/rewards/mean" in row]
    if not rewards:
        raise ValueError(f"{path} contains no critic/rewards/mean step records")
    peak = max(rewards)
    peak_step = next(
        step for step, row in records.items() if row.get("critic/rewards/mean") == peak
    )
    cumulative_time = 0.0
    cumulative_tokens = 0
    peak_time = 0.0
    peak_tokens = 0
    for step, row in records.items():
        cumulative_time += row.get("perf/time_per_step", row.get("timing_s/step", 0.0))
        cumulative_tokens += int(row.get("perf/total_num_tokens", 0))
        if step == peak_step:
            peak_time, peak_tokens = cumulative_time, cumulative_tokens
    return {
        "log_path": str(Path(path).resolve()),
        "updates": max(records, default=0),
        "thresholds": compute_threshold_stats(records, thresholds, window),
        "peak": {
            "reward": peak,
            "step": peak_step,
            "cumulative_time_s": peak_time,
            "cumulative_tokens": peak_tokens,
        },
        "total_time_s": cumulative_time,
        "total_tokens": cumulative_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each configuration",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = {}
    for item in args.log:
        if "=" not in item:
            parser.error(f"invalid --log {item!r}; expected LABEL=PATH")
        label, path = item.split("=", 1)
        result[label] = summarize(path, args.thresholds, args.window, args.max_step)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
