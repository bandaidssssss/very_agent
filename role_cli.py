#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import AgentSet
from config_utils import load_json, write_json


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one verl tuning agent role")
    parser.add_argument("--role", choices=["proposal", "feasibility", "diagnosis"], required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--output")
    parser.add_argument("--rules-only", action="store_true")
    args = parser.parse_args()

    agents = AgentSet(ROOT, "rules" if args.rules_only else "llm")
    context = load_json(args.context)
    if args.role == "proposal":
        result = agents.propose(context)
    elif args.role == "feasibility":
        result = agents.review(context)
    else:
        result = agents.diagnose(context)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
