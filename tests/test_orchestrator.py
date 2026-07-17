from __future__ import annotations

import unittest

from orchestrator import determine_stage


def hardware_trial(trial_id: int, throughput: float, result: str = "success") -> dict:
    return {
        "trial_id": trial_id,
        "stage": "hardware_tuning",
        "result": result,
        "performance": {"throughput": {"mean": throughput}},
    }


def stability_trial(trial_id: int, reward: float, slope: float = 0.01, kl: float = 0.02) -> dict:
    return {
        "trial_id": trial_id,
        "stage": "stability_tuning",
        "result": "success",
        "stability": {
            "reward": {"mean": reward},
            "reward_slope": slope,
            "actor_ppo_kl": {"max": kl},
        },
    }


class OrchestratorStageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "min_hardware_trials": 2,
            "max_hardware_trials": 6,
            "plateau_rounds": 2,
            "min_throughput_improvement": 0.02,
            "min_stability_trials": 2,
            "max_stability_trials": 4,
            "reward_collapse_slope": -0.01,
            "kl_warning": 0.1,
        }

    def test_initial_and_failed_hardware_stage(self) -> None:
        self.assertEqual(determine_stage([], self.config), "hardware_tuning")
        failed = {"trial_id": 1, "stage": "hardware_tuning", "result": "fail"}
        self.assertEqual(determine_stage([failed], self.config), "hardware_repair")

    def test_plateau_moves_to_stability(self) -> None:
        trials = [
            hardware_trial(1, 100),
            hardware_trial(2, 110),
            hardware_trial(3, 111),
            hardware_trial(4, 109),
        ]
        self.assertEqual(determine_stage(trials, self.config), "stability_tuning")

    def test_two_healthy_stability_trials_move_to_confirm(self) -> None:
        trials = [hardware_trial(index, 100 + index) for index in range(1, 7)]
        trials.extend([stability_trial(7, 0.1), stability_trial(8, 0.2)])
        self.assertEqual(determine_stage(trials, self.config), "confirm")


if __name__ == "__main__":
    unittest.main()
