from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PHASES = ("rollout", "actor_log_prob", "ref_log_prob", "training")


def _number(parameters: Mapping[str, Any], key: str, default: float) -> float:
    value = parameters.get(key, default)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _enabled(parameters: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = parameters.get(key, default)
    return bool(value) if isinstance(value, bool) else default


def _recompute_factor(parameters: Mapping[str, Any]) -> float:
    value = parameters.get(
        "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity"
    )
    if value in (None, "none", "None", False):
        return 1.0
    return 0.66 if str(value).lower() == "full" else 0.78


def _phase_pressure(phase: str, parameters: Mapping[str, Any]) -> float:
    prompt = _number(parameters, "data.max_prompt_length", 1024)
    response = _number(parameters, "data.max_response_length", 4096)
    sequence = max(1.0, prompt + response)
    rollout_tp = max(1.0, _number(parameters, "actor_rollout_ref.rollout.tensor_model_parallel_size", 1))
    actor_tp = max(1.0, _number(parameters, "actor_rollout_ref.actor.megatron.tensor_model_parallel_size", 1))
    actor_pp = max(1.0, _number(parameters, "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size", 1))
    ref_tp = max(1.0, _number(parameters, "actor_rollout_ref.ref.megatron.tensor_model_parallel_size", 1))
    ref_pp = max(1.0, _number(parameters, "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size", 1))

    if phase == "rollout":
        utilization = _number(parameters, "actor_rollout_ref.rollout.gpu_memory_utilization", 0.6)
        batched_tokens = _number(parameters, "actor_rollout_ref.rollout.max_num_batched_tokens", 65536)
        max_seqs = _number(parameters, "actor_rollout_ref.rollout.max_num_seqs", 256)
        rollout_n = _number(parameters, "actor_rollout_ref.rollout.n", 1)
        return (
            utilization**0.58
            * batched_tokens**0.14
            * max_seqs**0.10
            * sequence**0.08
            * rollout_n**0.03
            * rollout_tp**-0.07
        )

    if phase == "actor_log_prob":
        micro = _number(parameters, "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", 1)
        pressure = micro**0.42 * sequence**0.28 * actor_tp**-0.22
        if _enabled(parameters, "actor_rollout_ref.actor.megatron.use_remove_padding", False):
            pressure *= 0.86
        if _enabled(parameters, "actor_rollout_ref.actor.use_dynamic_bsz", False):
            pressure *= 0.94
        if _enabled(parameters, "actor_rollout_ref.actor.megatron.param_offload", False):
            pressure *= 0.82
        return pressure

    if phase == "ref_log_prob":
        micro = _number(parameters, "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", 1)
        pressure = micro**0.42 * sequence**0.28 * (ref_tp * ref_pp) ** -0.22
        if _enabled(parameters, "actor_rollout_ref.ref.megatron.sequence_parallel", False):
            pressure *= 0.93
        if _enabled(parameters, "actor_rollout_ref.ref.megatron.param_offload", False):
            pressure *= 0.76
        return pressure

    micro = _number(parameters, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 1)
    pressure = micro**0.40 * sequence**0.27 * (actor_tp * actor_pp) ** -0.23
    if _enabled(parameters, "actor_rollout_ref.actor.megatron.sequence_parallel", False):
        pressure *= 0.91
    if _enabled(parameters, "actor_rollout_ref.actor.megatron.use_distributed_optimizer", False):
        pressure *= 0.84
    if _enabled(parameters, "actor_rollout_ref.actor.megatron.optimizer_offload", False):
        pressure *= 0.78
    if _enabled(parameters, "actor_rollout_ref.actor.megatron.param_offload", False):
        pressure *= 0.78
    return pressure * _recompute_factor(parameters)


def _phase_peaks(trial: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    memory = trial.get("memory_by_phase_pct")
    if not isinstance(memory, Mapping):
        return result
    for phase in PHASES:
        value = memory.get(phase)
        if isinstance(value, Mapping):
            value = value.get("max")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[phase] = float(value)
    return result


def _same_parameters(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(right, sort_keys=True, default=str)


def _reference_trial(
    current: Mapping[str, Any], trials: Sequence[Mapping[str, Any]], reference_trial_id: int | None
) -> Mapping[str, Any] | None:
    observed = [trial for trial in trials if _phase_peaks(trial) and isinstance(trial.get("parameters"), Mapping)]
    if reference_trial_id is not None:
        return next((trial for trial in observed if trial.get("trial_id") == reference_trial_id), None)
    exact = [trial for trial in observed if _same_parameters(trial.get("parameters", {}), current)]
    return (exact or observed)[-1] if observed else None


def estimate_phase_memory(
    current_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]] = (),
    memory_limit_pct: float = 92.0,
    reference_trial_id: int | None = None,
) -> dict[str, Any]:
    reference = _reference_trial(current_parameters, trials, reference_trial_id)
    reference_parameters: Mapping[str, Any] = current_parameters
    peaks: dict[str, float] = {}
    if reference:
        reference_parameters = reference.get("parameters", current_parameters)
        peaks = _phase_peaks(reference)

    phases: dict[str, Any] = {}
    for phase in PHASES:
        denominator = max(_phase_pressure(phase, reference_parameters), 1e-12)
        ratio = max(0.2, min(5.0, _phase_pressure(phase, candidate_parameters) / denominator))
        projected = peaks.get(phase)
        projected = projected * ratio if projected is not None else None
        if projected is None:
            risk = "unknown_without_observed_anchor"
            headroom = None
        else:
            headroom = memory_limit_pct - projected
            risk = "high" if projected >= memory_limit_pct else ("watch" if headroom < 5.0 else "low")
        phases[phase] = {
            "reference_pct": round(peaks[phase], 2) if phase in peaks else None,
            "pressure_ratio": round(ratio, 3),
            "projected_pct": round(projected, 2) if projected is not None else None,
            "headroom_to_limit_pct": round(headroom, 2) if headroom is not None else None,
            "risk": risk,
        }

    return {
        "method": "empirical_phase_relative" if reference else "relative_pressure_only",
        "confidence": "medium" if reference else "low",
        "memory_limit_pct": memory_limit_pct,
        "reference_trial_id": reference.get("trial_id") if reference else None,
        "phases": phases,
        "limitations": [
            "The estimator projects relative pressure from parameter changes; it is not a tensor-allocation simulator.",
            "Absolute percentages require a prior trial with phase-tagged GPU memory observations.",
            "Live SMI snapshots cannot replace rollout/actor/ref/training phase samples.",
            "A real short resource-gate trial remains the final memory authority.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate verl phase memory from an empirical reference trial")
    parser.add_argument("--current", required=True, help="Current parameter JSON")
    parser.add_argument("--candidate", required=True, help="Candidate parameter JSON")
    parser.add_argument("--trials", help="Optional JSON array or JSONL trial history")
    parser.add_argument("--memory-limit-pct", type=float, default=92.0)
    args = parser.parse_args()

    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    trials = []
    if args.trials:
        text = Path(args.trials).read_text(encoding="utf-8").strip()
        trials = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line]
    print(json.dumps(estimate_phase_memory(current, candidate, trials, args.memory_limit_pct), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
