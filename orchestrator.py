from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

from agents import AgentError, AgentSet
from config_utils import append_jsonl, apply_changes, read_jsonl, write_json
from runner import run_trial
from validator import editable_parameters, validate_candidate


def _metric_mean(trial: Mapping[str, Any], *path: str) -> float | None:
    value: Any = trial
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if isinstance(value, Mapping):
        value = value.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _hardware_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in trials if str(trial.get("stage", "")).startswith("hardware")]


def _stability_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in trials if trial.get("stage") == "stability_tuning"]


def _confirm_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in trials if trial.get("stage") == "confirm"]


def _successful(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in trials if trial.get("result") == "success"]


def best_hardware_trial(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for trial in _successful(_hardware_trials(trials)):
        throughput = _metric_mean(trial, "performance", "throughput")
        if throughput is not None:
            candidates.append((throughput, trial))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def best_stability_trial(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for trial in _successful(_stability_trials(trials)):
        reward = _metric_mean(trial, "stability", "reward")
        if reward is not None:
            candidates.append((reward, trial))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def hardware_plateaued(trials: list[dict[str, Any]], config: Mapping[str, Any]) -> bool:
    successful = _successful(_hardware_trials(trials))
    minimum = int(config.get("min_hardware_trials", 2))
    plateau_rounds = int(config.get("plateau_rounds", 2))
    if len(successful) < minimum + plateau_rounds:
        return False
    scores = [_metric_mean(trial, "performance", "throughput") for trial in successful]
    scores = [score for score in scores if score is not None]
    if len(scores) < minimum + plateau_rounds:
        return False
    threshold = float(config.get("min_throughput_improvement", 0.02))
    best_before = max(scores[:-plateau_rounds])
    return all(score <= best_before * (1.0 + threshold) for score in scores[-plateau_rounds:])


def stability_healthy(trial: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if trial.get("result") != "success":
        return False
    slope = trial.get("stability", {}).get("reward_slope")
    kl_max = trial.get("stability", {}).get("actor_ppo_kl", {}).get("max")
    if slope is not None and slope < float(config.get("reward_collapse_slope", -0.01)):
        return False
    if kl_max is not None and kl_max > float(config.get("kl_warning", 0.1)):
        return False
    return True


def determine_stage(trials: list[dict[str, Any]], config: Mapping[str, Any]) -> str:
    if _confirm_trials(trials):
        return "done"
    hardware = _hardware_trials(trials)
    successful_hardware = _successful(hardware)
    if not successful_hardware:
        return "hardware_repair" if hardware else "hardware_tuning"
    if len(hardware) < int(config.get("min_hardware_trials", 2)):
        return "hardware_tuning"
    if len(hardware) < int(config.get("max_hardware_trials", 6)) and not hardware_plateaued(trials, config):
        return "hardware_tuning"

    stability = _stability_trials(trials)
    healthy = [trial for trial in stability if stability_healthy(trial, config)]
    if len(stability) >= int(config.get("max_stability_trials", 4)):
        return "confirm" if healthy else "stopped_unstable"
    if len(healthy) >= int(config.get("min_stability_trials", 2)):
        return "confirm"
    return "stability_tuning"


def trial_budget(stage: str, config: Mapping[str, Any]) -> int:
    if stage.startswith("hardware"):
        return int(config.get("hardware_trial_updates", 20))
    if stage == "stability_tuning":
        return int(config.get("stability_trial_updates", 80))
    if stage == "confirm":
        return int(config.get("confirm_trial_updates", 300))
    return 0


def _compact_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "trial_id",
        "stage",
        "result",
        "updates_completed",
        "parameters",
        "error",
        "resource",
        "memory_by_phase_pct",
        "performance",
        "stability",
        "diagnosis",
        "failure_phase",
    ]
    return {key: copy.deepcopy(trial[key]) for key in keys if key in trial}


class TuningOrchestrator:
    def __init__(
        self,
        root: str | Path,
        base_parameters: Mapping[str, Any],
        agent_config: Mapping[str, Any],
    ) -> None:
        self.root = Path(root)
        self.base_parameters = dict(base_parameters)
        self.config = dict(agent_config)
        self.output_dir = Path(os.getenv("OUTPUT_PATH", str(self.config["output_dir"]))).expanduser().resolve()
        self.history_path = self.output_dir / "trials.jsonl"
        self.state_path = self.output_dir / "state.json"
        self.agents = AgentSet(self.root, str(self.config.get("agent_mode", "llm")))

    def trials(self) -> list[dict[str, Any]]:
        return read_jsonl(self.history_path)

    def _starting_parameters(self, stage: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
        if stage.startswith("hardware"):
            best = best_hardware_trial(trials)
            if best:
                return dict(best["parameters"])
            if trials:
                return dict(trials[-1]["parameters"])
            return dict(self.base_parameters)
        if stage == "stability_tuning":
            best = best_stability_trial(trials) or best_hardware_trial(trials)
            if not best:
                raise RuntimeError("stability tuning requires a successful hardware trial")
            return dict(best["parameters"])
        if stage == "confirm":
            best = best_stability_trial(trials) or best_hardware_trial(trials)
            if not best:
                raise RuntimeError("confirmation requires a successful candidate")
            return dict(best["parameters"])
        raise RuntimeError(f"unsupported stage: {stage}")

    def _diagnosis(self, trials: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not trials or trials[-1].get("result") == "success":
            return None
        context = {"trial": _compact_trial(trials[-1])}
        return self.agents.diagnose(context)

    def _propose_candidate(
        self, stage: str, current: Mapping[str, Any], trials: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        diagnosis = self._diagnosis(trials)
        rejections: list[dict[str, Any]] = []
        max_rounds = int(self.config.get("max_validation_rounds", 3))
        for _ in range(max_rounds):
            context = {
                "current_stage": stage,
                "mode": "failure_repair" if diagnosis else stage,
                "current_parameters": dict(current),
                "editable_parameters": editable_parameters(stage),
                "constraints": {
                    "max_parameter_changes": self.config.get("max_parameter_changes", 3),
                    "preserve_hardware_token_budget": self.config.get("preserve_hardware_token_budget", True),
                    "resource_memory_limit_pct": self.config.get("resource_memory_limit_pct", 95.0),
                    "throughput_memory_limit_pct": self.config.get("throughput_memory_limit_pct", 92.0),
                },
                "diagnosis": diagnosis,
                "recent_trials": [_compact_trial(trial) for trial in trials[-8:]],
                "rejected_attempts": rejections,
            }
            proposal = self.agents.propose(context)
            decision = proposal.get("decision")
            if decision in {"keep", "stop"}:
                return dict(current), proposal, {"verdict": "valid", "reason": "no parameter change"}
            changes = proposal.get("changes")
            if not isinstance(changes, Mapping) or not changes:
                rejections.append({"proposal": proposal, "violations": ["changes must be a non-empty object"]})
                continue
            candidate = apply_changes(current, changes)
            validation = validate_candidate(
                candidate, changes, stage, self.config, self.base_parameters, trials
            )
            if not validation.valid:
                rejections.append({"proposal": proposal, **validation.as_dict()})
                continue
            review = self.agents.review(
                {
                    "current_stage": stage,
                    "current_parameters": dict(current),
                    "candidate_parameters": candidate,
                    "changes": dict(changes),
                    "proposal_reason": proposal.get("reason"),
                    "last_trial": _compact_trial(trials[-1]) if trials else None,
                    "diagnosis": diagnosis,
                    "memory_limits": {
                        "resource": self.config.get("resource_memory_limit_pct", 95.0),
                        "throughput": self.config.get("throughput_memory_limit_pct", 92.0),
                    },
                }
            )
            if review.get("verdict") == "valid":
                return candidate, proposal, review
            rejections.append({"proposal": proposal, "feasibility": review})
        raise AgentError(f"no feasible proposal after {max_rounds} rounds: {rejections}")

    def run(self, max_trials: int = 1, dry_run: bool = False) -> list[dict[str, Any]]:
        produced = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(max_trials):
            trials = self.trials()
            stage = determine_stage(trials, self.config)
            if stage in {"done", "stopped_unstable"}:
                write_json(
                    self.state_path,
                    {
                        "current_stage": stage,
                        "last_trial_id": len(trials),
                        "history_path": str(self.history_path),
                    },
                )
                break
            trial_id = len(trials) + 1
            parameters = self._starting_parameters(stage, trials)
            proposal: dict[str, Any] = {"decision": "keep", "reason": "stage baseline"}
            review: dict[str, Any] = {"verdict": "valid", "reason": "stage baseline"}

            first_hardware = not _hardware_trials(trials)
            first_stability = stage == "stability_tuning" and not _stability_trials(trials)
            if not first_hardware and not first_stability and stage != "confirm":
                parameters, proposal, review = self._propose_candidate(stage, parameters, trials)
                if proposal.get("decision") in {"keep", "stop"}:
                    if stage.startswith("hardware") and best_hardware_trial(trials):
                        stage = "stability_tuning"
                        parameters = dict(best_hardware_trial(trials)["parameters"])
                    elif stage == "stability_tuning" and best_stability_trial(trials):
                        stage = "confirm"
                        parameters = dict(best_stability_trial(trials)["parameters"])
                    else:
                        write_json(
                            self.state_path,
                            {
                                "current_stage": "stopped_no_candidate",
                                "last_trial_id": len(trials),
                                "history_path": str(self.history_path),
                                "proposal": proposal,
                            },
                        )
                        break

            report = run_trial(
                parameters,
                self.config,
                trial_id,
                stage,
                trial_budget(stage, self.config),
                dry_run=dry_run,
            )
            report["proposal"] = proposal
            report["feasibility"] = review
            if not dry_run:
                append_jsonl(self.history_path, report)
                if stage == "confirm":
                    write_json(self.output_dir / "final_result.json", report)
            produced.append(report)
            write_json(
                self.state_path,
                {
                    "current_stage": stage if dry_run else determine_stage(self.trials(), self.config),
                    "last_trial_id": trial_id,
                    "history_path": str(self.history_path),
                },
            )
            if dry_run:
                break
        return produced
