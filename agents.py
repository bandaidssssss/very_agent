from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


class AgentError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S)
    if fenced:
        stripped = fenced.group(1)
    else:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise AgentError(f"agent did not return valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AgentError("agent response must be a JSON object")
    return value


class LLMRoleAgent:
    def __init__(self, prompt_path: str | Path) -> None:
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        self.api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("INFER_MODEL", "gpt-5")

    def run(self, context: Mapping[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise AgentError("API_KEY or OPENAI_API_KEY is required for LLM agent decisions")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentError("the openai package is required for LLM agent decisions") from exc
        kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": 120.0, "max_retries": 2}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
            ],
            max_output_tokens=4096,
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            chunks = []
            for item in getattr(response, "output", []):
                for content in getattr(item, "content", []):
                    text = getattr(content, "text", None)
                    if text:
                        chunks.append(text)
            output_text = "\n".join(chunks)
        return _extract_json(output_text or "")


class AgentSet:
    def __init__(self, root: str | Path, mode: str = "llm") -> None:
        prompt_root = Path(root) / "prompts"
        self.mode = mode
        self.proposal = LLMRoleAgent(prompt_root / "proposal.md")
        self.feasibility = LLMRoleAgent(prompt_root / "feasibility.md")
        self.diagnosis = LLMRoleAgent(prompt_root / "diagnosis.md")

    def propose(self, context: Mapping[str, Any]) -> dict[str, Any]:
        if self.mode == "rules":
            return {"decision": "keep", "reason": "rules mode keeps the current configuration", "changes": {}}
        return self.proposal.run(context)

    def review(self, context: Mapping[str, Any]) -> dict[str, Any]:
        if self.mode == "rules":
            return {"verdict": "valid", "reason": "deterministic validation passed", "risks": []}
        return self.feasibility.run(context)

    def diagnose(self, context: Mapping[str, Any]) -> dict[str, Any]:
        if self.mode == "rules":
            error_type = context.get("trial", {}).get("error", {}).get("type") or "UNKNOWN_FAILURE"
            return {
                "failure_type": error_type,
                "training_substage": context.get("trial", {}).get("failure_phase")
                or context.get("trial", {}).get("resource", {}).get("memory_bottleneck")
                or "unknown",
                "evidence": context.get("trial", {}).get("error", {}).get("evidence", []),
                "reason": "deterministic log classification",
                "confidence": 1.0 if error_type != "UNKNOWN_FAILURE" else 0.3,
            }
        return self.diagnosis.run(context)
