from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from runner import GPUSampler, HealthAgentWorker, PhaseTracker


class GPUSamplerTest(unittest.TestCase):
    def test_nvidia_csv_query(self) -> None:
        response = subprocess.CompletedProcess([], 0, "0, 1024, 8192, 75\n", "")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"PLATFORM": "A100", "GPU_SMI": ""}
        ), mock.patch("runner.shutil.which", return_value="/usr/bin/nvidia-smi"), mock.patch(
            "runner.subprocess.run", return_value=response
        ):
            sampler = GPUSampler(Path(directory) / "gpu.csv", PhaseTracker(), 1.0, "A100")
            rows = sampler._query_rows()
        self.assertEqual(rows, [["0", "1024", "8192", "75"]])

    def test_v5000_human_table_fallback(self) -> None:
        query = subprocess.CompletedProcess([], 1, "", "unsupported query")
        table = subprocess.CompletedProcess([], 0, "0 Default a b c d e f 1024 x 2048 y 50\n", "")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"PLATFORM": "V5000", "GPU_SMI": ""}
        ), mock.patch("runner.shutil.which", return_value="/usr/bin/xpu-smi"), mock.patch(
            "runner.subprocess.run", side_effect=[query, table]
        ):
            sampler = GPUSampler(Path(directory) / "gpu.csv", PhaseTracker(), 1.0, "V5000")
            rows = sampler._query_rows()
        self.assertEqual(rows, [["0", "1024", "2048", "50"]])


class HealthAgentWorkerTest(unittest.TestCase):
    def test_agent_failure_is_returned_without_raising_in_caller(self) -> None:
        def fail(_context):
            raise RuntimeError("agent unavailable")

        worker = HealthAgentWorker(fail)
        self.assertTrue(worker.submit("event-1", {"health_event": {}}))
        result = None
        for _ in range(100):
            result = worker.poll()
            if result is not None:
                break
            time.sleep(0.001)
        self.assertIsNotNone(result)
        self.assertFalse((result or {}).get("ok"))
        self.assertIn("agent unavailable", (result or {}).get("error", ""))


if __name__ == "__main__":
    unittest.main()
