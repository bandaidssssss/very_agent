from __future__ import annotations

import math
import re
from collections import deque
from typing import Any, Mapping


NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
PAIR_RE = re.compile(rf"([^\s:]+):({NUMBER})")
STEP_RE = re.compile(r"step:(\d+)")


PAPER_KL_GROWTH_THRESHOLD = 0.15
PAPER_REWARD_DROP_THRESHOLD = 0.10
PAPER_CONSECUTIVE_STEPS = 5


def parse_online_step(line: str) -> tuple[int, dict[str, float]] | None:
    """Parse one verl step summary without waiting for the trial to finish."""
    step_match = STEP_RE.search(line)
    if not step_match:
        return None
    metrics = {key: float(value) for key, value in PAIR_RE.findall(line)}
    if not metrics:
        return None
    return int(step_match.group(1)), metrics


class OnlineHealthMonitor:
    """Faithful online implementation of the JF-HPO early-stop predicates.

    The monitor only raises structured events. It never stops a process and never
    calls an Agent. This keeps the paper rules deterministic and independently
    testable while a separate decision layer reviews each trigger.
    """

    def __init__(
        self,
        *,
        kl_metric: str = "actor/kl_loss",
        reward_metric: str = "critic/rewards/mean",
        kl_growth_threshold: float = PAPER_KL_GROWTH_THRESHOLD,
        reward_drop_threshold: float = PAPER_REWARD_DROP_THRESHOLD,
        consecutive_steps: int = PAPER_CONSECUTIVE_STEPS,
        reward_zero_epsilon: float = 0.0,
        ratio_epsilon: float = 1e-12,
        warmup_updates: int = 0,
        cooldown_updates: int | None = None,
    ) -> None:
        if kl_growth_threshold < 0 or reward_drop_threshold < 0:
            raise ValueError("health thresholds must be non-negative")
        if consecutive_steps < 1:
            raise ValueError("consecutive_steps must be positive")
        self.kl_metric = kl_metric
        self.reward_metric = reward_metric
        self.kl_growth_threshold = float(kl_growth_threshold)
        self.reward_drop_threshold = float(reward_drop_threshold)
        self.consecutive_steps = int(consecutive_steps)
        self.reward_zero_epsilon = float(reward_zero_epsilon)
        self.ratio_epsilon = max(float(ratio_epsilon), 1e-30)
        self.warmup_updates = max(0, int(warmup_updates))
        self.cooldown_updates = max(
            1,
            int(cooldown_updates if cooldown_updates is not None else consecutive_steps),
        )

        self.previous_kl: float | None = None
        self.previous_reward: float | None = None
        self.counters = {
            "kl_growth": 0,
            "reward_drop": 0,
            "reward_zero": 0,
        }
        self.last_step = 0
        self.last_trigger_step: int | None = None
        self.trigger_count = 0
        self.recent_steps: deque[dict[str, Any]] = deque(maxlen=max(10, 2 * self.consecutive_steps))

    @staticmethod
    def _finite(value: Any) -> float | None:
        if not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def _ratio(self, delta: float, previous: float) -> float:
        return delta / max(abs(previous), self.ratio_epsilon)

    def _advance_counter(self, name: str, active: bool) -> int:
        self.counters[name] = self.counters[name] + 1 if active else 0
        return self.counters[name]

    def add_step(self, step: int, metrics: Mapping[str, float]) -> dict[str, Any] | None:
        kl = self._finite(metrics.get(self.kl_metric))
        reward = self._finite(metrics.get(self.reward_metric))
        kl_growth_ratio = None
        reward_drop_ratio = None

        if kl is not None and self.previous_kl is not None:
            kl_growth_ratio = self._ratio(kl - self.previous_kl, self.previous_kl)
        if reward is not None and self.previous_reward is not None:
            reward_drop_ratio = self._ratio(self.previous_reward - reward, self.previous_reward)

        eligible = step > self.warmup_updates
        kl_active = bool(
            eligible
            and kl_growth_ratio is not None
            and kl_growth_ratio > self.kl_growth_threshold
        )
        reward_drop_active = bool(
            eligible
            and reward_drop_ratio is not None
            and reward_drop_ratio > self.reward_drop_threshold
        )
        reward_zero_active = bool(
            eligible
            and reward is not None
            and abs(reward) <= self.reward_zero_epsilon
        )

        self._advance_counter("kl_growth", kl_active)
        self._advance_counter("reward_drop", reward_drop_active)
        self._advance_counter("reward_zero", reward_zero_active)
        self.last_step = max(self.last_step, int(step))

        selected_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if key
            in {
                self.kl_metric,
                self.reward_metric,
                "actor/ppo_kl",
                "actor/entropy",
                "actor/grad_norm",
                "actor/pg_clipfrac",
                "response_length/mean",
                "response_length/clip_ratio",
                "response/aborted_ratio",
                "perf/max_memory_allocated_gb",
                "perf/max_memory_reserved_gb",
                "perf/cpu_memory_used_gb",
                "perf/throughput",
                "perf/time_per_step",
            }
        }
        row = {
            "step": int(step),
            "metrics": selected_metrics,
            "derived": {
                "kl_growth_ratio": kl_growth_ratio,
                "reward_drop_ratio": reward_drop_ratio,
            },
            "counters": dict(self.counters),
        }
        self.recent_steps.append(row)

        if kl is not None:
            self.previous_kl = kl
        if reward is not None:
            self.previous_reward = reward

        rules = []
        if self.counters["kl_growth"] >= self.consecutive_steps:
            rules.append(
                {
                    "name": "jf_hpo_kl_growth",
                    "metric": self.kl_metric,
                    "observed_ratio": kl_growth_ratio,
                    "threshold": self.kl_growth_threshold,
                    "consecutive_steps": self.counters["kl_growth"],
                }
            )
        if self.counters["reward_drop"] >= self.consecutive_steps:
            rules.append(
                {
                    "name": "jf_hpo_reward_drop",
                    "metric": self.reward_metric,
                    "observed_ratio": reward_drop_ratio,
                    "threshold": self.reward_drop_threshold,
                    "consecutive_steps": self.counters["reward_drop"],
                }
            )
        if self.counters["reward_zero"] >= self.consecutive_steps:
            rules.append(
                {
                    "name": "jf_hpo_reward_zero",
                    "metric": self.reward_metric,
                    "observed_value": reward,
                    "epsilon": self.reward_zero_epsilon,
                    "consecutive_steps": self.counters["reward_zero"],
                }
            )
        if not rules:
            return None

        if self.last_trigger_step is not None and step - self.last_trigger_step < self.cooldown_updates:
            return None

        self.last_trigger_step = int(step)
        self.trigger_count += 1
        return {
            "source": "JF-HPO",
            "step": int(step),
            "severity": "high" if len(rules) > 1 else "warning",
            "rules": rules,
            "paper_parameters": {
                "tau_1_kl_growth": self.kl_growth_threshold,
                "tau_2_reward_drop": self.reward_drop_threshold,
                "k_consecutive_steps": self.consecutive_steps,
            },
            "recent_steps": list(self.recent_steps),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "kl_metric": self.kl_metric,
            "reward_metric": self.reward_metric,
            "paper_parameters": {
                "tau_1_kl_growth": self.kl_growth_threshold,
                "tau_2_reward_drop": self.reward_drop_threshold,
                "k_consecutive_steps": self.consecutive_steps,
            },
            "warmup_updates": self.warmup_updates,
            "cooldown_updates": self.cooldown_updates,
            "last_step": self.last_step,
            "trigger_count": self.trigger_count,
            "counters": dict(self.counters),
        }
