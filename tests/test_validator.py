from __future__ import annotations

import unittest
from pathlib import Path

from verl_agent.config_utils import load_json
from verl_agent.validator import validate_candidate


class ValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.base = load_json(root / "config" / "base_parameters.json")
        cls.config = load_json(root / "config" / "agent_config.json")

    def test_valid_micro_batch_change(self) -> None:
        candidate = dict(self.base)
        candidate["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] = 2
        result = validate_candidate(
            candidate,
            {"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2},
            "hardware_tuning",
            self.config,
            self.base,
            [],
        )
        self.assertTrue(result.valid, result.violations)

    def test_rejects_stability_hardware_change(self) -> None:
        candidate = dict(self.base)
        candidate["actor_rollout_ref.rollout.max_num_seqs"] = 128
        result = validate_candidate(
            candidate,
            {"actor_rollout_ref.rollout.max_num_seqs": 128},
            "stability_tuning",
            self.config,
            self.base,
            [],
        )
        self.assertFalse(result.valid)

    def test_rejects_changed_hardware_token_budget(self) -> None:
        candidate = dict(self.base)
        candidate["actor_rollout_ref.rollout.n"] = 8
        result = validate_candidate(
            candidate,
            {"actor_rollout_ref.rollout.n": 8},
            "hardware_tuning",
            self.config,
            self.base,
            [],
        )
        self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()
