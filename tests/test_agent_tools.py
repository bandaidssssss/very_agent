from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_tools import ToolRegistry
from config_utils import load_json


ROOT = Path(__file__).resolve().parents[1]


class AgentToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_root = Path(self.temp_dir.name)
        self.history_path = self.temp_root / "output" / "trials.jsonl"
        self.history_path.parent.mkdir(parents=True)
        self.config = load_json(ROOT / "config" / "agent_config.json")
        self.base = load_json(ROOT / "config" / "base_parameters.json")

    def registry(self, **overrides: object) -> ToolRegistry:
        config = {**self.config, **overrides}
        return ToolRegistry(ROOT, config, self.history_path)

    def test_parameter_understanding_is_allowlisted(self) -> None:
        registry = self.registry()
        runtime = registry.runtime({})
        result = registry.execute(
            "proposal",
            "parameter_understanding",
            {"items": ["actor_rollout_ref.rollout.gpu_memory_utilization", "missing.parameter"]},
            runtime,
        )
        self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization", result["parameters"])
        self.assertEqual(result["unknown_parameters"], ["missing.parameter"])
        with self.assertRaises(RuntimeError):
            registry.execute("proposal", "read_trial_log_excerpt", {"trial_id": 1}, runtime)

    def test_memory_estimator_uses_phase_observation_anchor(self) -> None:
        trial = {
            "trial_id": 1,
            "parameters": self.base,
            "memory_by_phase_pct": {
                "rollout": {"max": 70.0},
                "actor_log_prob": {"max": 60.0},
                "ref_log_prob": {"max": 55.0},
                "training": {"max": 72.0},
            },
        }
        context = {
            "current_parameters": self.base,
            "recent_trials": [trial],
            "constraints": {"throughput_memory_limit_pct": 92.0},
        }
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "memory_estimator",
            {"changes": {"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2}},
            registry.runtime(context),
        )
        self.assertEqual(result["method"], "empirical_phase_relative")
        self.assertEqual(result["reference_trial_id"], 1)
        self.assertGreater(result["phases"]["training"]["projected_pct"], 72.0)
        self.assertAlmostEqual(result["phases"]["rollout"]["projected_pct"], 70.0)

    def test_search_verl_docs_is_bounded_to_configured_root(self) -> None:
        source = self.temp_root / "verl" / "workers" / "config" / "actor.py"
        source.parent.mkdir(parents=True)
        source.write_text("ppo_micro_batch_size_per_gpu controls local batches\n", encoding="utf-8")
        registry = self.registry(verl_root=str(self.temp_root))
        result = registry.execute(
            "proposal",
            "search_verl_docs",
            {"query": "ppo_micro_batch_size_per_gpu", "max_results": 3},
            registry.runtime({}),
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["matches"][0]["path"], "verl/workers/config/actor.py")

    def test_trial_history_query_can_return_successful_parameters(self) -> None:
        rows = [
            {"trial_id": 1, "stage": "hardware_tuning", "result": "fail", "parameters": {"x": 1}},
            {
                "trial_id": 2,
                "stage": "hardware_tuning",
                "result": "success",
                "parameters": {"x": 2},
                "performance": {"throughput": {"mean": 12.0}},
            },
        ]
        self.history_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "query_trial_history",
            {"result": "success", "sort_by": "throughput", "include_parameters": True},
            registry.runtime({}),
        )
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["trials"][0]["parameters"], {"x": 2})


if __name__ == "__main__":
    unittest.main()
