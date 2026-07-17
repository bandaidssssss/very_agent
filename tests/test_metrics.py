from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from verl_agent.metrics import analyze_trial, compute_threshold_stats, parse_step_records


class MetricsTest(unittest.TestCase):
    def test_parses_steps_and_phase_memory(self) -> None:
        text = """
Before generate_sequences, memory allocated (GB): 2.0, memory reserved (GB): 3.0, device memory used/total (GB): 32.0/64.0
After generate_sequences, memory allocated (GB): 2.0, memory reserved (GB): 3.0, device memory used/total (GB): 48.0/64.0
Before compute_log_prob, memory allocated (GB): 2.0, memory reserved (GB): 3.0, device memory used/total (GB): 50.0/64.0
After compute_log_prob, memory allocated (GB): 2.0, memory reserved (GB): 3.0, device memory used/total (GB): 58.0/64.0
step:1 - critic/rewards/mean:-0.2 - actor/ppo_kl:0.01 - actor/entropy:0.3 - actor/pg_loss:0.02 - actor/pg_clipfrac:0.1 - timing_s/gen:10 - timing_s/old_log_prob:4 - timing_s/ref:2 - timing_s/update_actor:8 - perf/time_per_step:24 - perf/total_num_tokens:1000 - perf/throughput:5
step:2 - critic/rewards/mean:0.2 - actor/ppo_kl:0.02 - actor/entropy:0.2 - actor/pg_loss:0.01 - actor/pg_clipfrac:0.2 - timing_s/gen:9 - timing_s/old_log_prob:4 - timing_s/ref:2 - timing_s/update_actor:7 - perf/time_per_step:22 - perf/total_num_tokens:1100 - perf/throughput:6
""".strip()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(text, encoding="utf-8")
            report = analyze_trial(path, None, warmup_updates=0)
        self.assertEqual(report["updates_completed"], 2)
        self.assertEqual(report["performance"]["time_bottleneck"], "rollout")
        self.assertAlmostEqual(report["memory_by_phase_pct"]["actor_log_prob"]["max"], 90.625)
        self.assertAlmostEqual(report["stability"]["reward_slope"], 0.4)

    def test_threshold_stats(self) -> None:
        records = {
            1: {"critic/rewards/mean": -0.1, "perf/time_per_step": 2, "perf/total_num_tokens": 10},
            2: {"critic/rewards/mean": 0.1, "perf/time_per_step": 3, "perf/total_num_tokens": 20},
            3: {"critic/rewards/mean": 0.2, "perf/time_per_step": 4, "perf/total_num_tokens": 30},
        }
        result = compute_threshold_stats(records, [0.0, 0.1], window=1)
        self.assertEqual(result["0.0"]["step"], 2)
        self.assertEqual(result["0.1"]["cumulative_tokens"], 30)

    def test_normal_nccl_configuration_is_not_failure(self) -> None:
        text = """
ray init kwargs: {'env_vars': {'NCCL_CUMEM_ENABLE': '0'}}
config: {'nccl_timeout': 600}
step:1 - critic/rewards/mean:0.1 - perf/time_per_step:2 - perf/total_num_tokens:10
""".strip()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(text, encoding="utf-8")
            report = analyze_trial(path, None, warmup_updates=0)
        self.assertEqual(report["result"], "success")


if __name__ == "__main__":
    unittest.main()
