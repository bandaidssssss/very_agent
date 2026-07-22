from __future__ import annotations

import unittest

from health_monitor import OnlineHealthMonitor, parse_online_step


class OnlineHealthMonitorTest(unittest.TestCase):
    def test_parse_verl_step_summary(self) -> None:
        parsed = parse_online_step(
            "step:7 - actor/kl_loss:0.01 - critic/rewards/mean:-0.25 - actor/entropy:0.3"
        )
        self.assertIsNotNone(parsed)
        step, metrics = parsed or (0, {})
        self.assertEqual(step, 7)
        self.assertEqual(metrics["critic/rewards/mean"], -0.25)

    def test_paper_kl_rule_triggers_after_five_consecutive_increases(self) -> None:
        monitor = OnlineHealthMonitor(cooldown_updates=5)
        trigger = None
        kl = 1.0
        for step in range(1, 7):
            trigger = monitor.add_step(
                step,
                {"actor/kl_loss": kl, "critic/rewards/mean": 1.0},
            )
            kl *= 1.2
        self.assertIsNotNone(trigger)
        rules = {row["name"] for row in (trigger or {}).get("rules", [])}
        self.assertEqual(rules, {"jf_hpo_kl_growth"})
        self.assertEqual((trigger or {})["paper_parameters"]["tau_1_kl_growth"], 0.15)
        self.assertEqual((trigger or {})["paper_parameters"]["k_consecutive_steps"], 5)

    def test_negative_reward_drop_uses_absolute_denominator(self) -> None:
        monitor = OnlineHealthMonitor(cooldown_updates=5)
        reward = -1.0
        trigger = None
        for step in range(1, 7):
            trigger = monitor.add_step(
                step,
                {"actor/kl_loss": 1.0, "critic/rewards/mean": reward},
            )
            reward *= 1.2
        self.assertIsNotNone(trigger)
        rules = {row["name"] for row in (trigger or {}).get("rules", [])}
        self.assertEqual(rules, {"jf_hpo_reward_drop"})

    def test_zero_reward_rule_is_independent(self) -> None:
        monitor = OnlineHealthMonitor(cooldown_updates=5)
        trigger = None
        for step in range(1, 6):
            trigger = monitor.add_step(
                step,
                {"actor/kl_loss": 1.0, "critic/rewards/mean": 0.0},
            )
        self.assertIsNotNone(trigger)
        rules = {row["name"] for row in (trigger or {}).get("rules", [])}
        self.assertEqual(rules, {"jf_hpo_reward_zero"})

    def test_intermittent_growth_resets_consecutive_counter(self) -> None:
        monitor = OnlineHealthMonitor(cooldown_updates=5)
        values = [1.0, 1.2, 1.44, 1.40, 1.68, 2.016, 2.4192]
        triggers = [
            monitor.add_step(
                step,
                {"actor/kl_loss": value, "critic/rewards/mean": 1.0},
            )
            for step, value in enumerate(values, 1)
        ]
        self.assertTrue(all(trigger is None for trigger in triggers))


if __name__ == "__main__":
    unittest.main()
