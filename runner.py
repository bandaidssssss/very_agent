from __future__ import annotations

import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from config_utils import hydra_overrides, write_json
from metrics import analyze_trial


STEP_RE = re.compile(r"step:(\d+)")
FATAL_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|ChildFailedError|DistBackendError|"
    r"NCCL[^\n]{0,80}(?:WARN|ERROR|unhandled|failed|failure)",
    re.I,
)
PHASE_START = {
    "Before generate_sequences": "rollout",
    "Before compute_log_prob": "actor_log_prob",
    "Before compute_ref_log_prob": "ref_log_prob",
    "Before update_actor": "training",
}
PHASE_END = tuple(value.replace("Before", "After", 1) for value in PHASE_START)


class PhaseTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase = "startup"

    def update_from_log(self, line: str) -> None:
        with self._lock:
            for marker, phase in PHASE_START.items():
                if marker in line:
                    self._phase = phase
                    return
            if any(marker in line for marker in PHASE_END):
                self._phase = "between_phases"

    def get(self) -> str:
        with self._lock:
            return self._phase


class GPUSampler(threading.Thread):
    def __init__(self, output_path: Path, tracker: PhaseTracker, interval: float, platform: str) -> None:
        super().__init__(daemon=True)
        self.output_path = output_path
        self.tracker = tracker
        self.interval = interval
        self.stop_event = threading.Event()
        self.max_memory_pct = 0.0
        self.platform = os.getenv("PLATFORM", platform).upper()
        configured_smi = os.getenv("GPU_SMI")
        default_smi = "xpu-smi" if self.platform == "V5000" else "nvidia-smi"
        self.executable = configured_smi or shutil.which(default_smi)

    @staticmethod
    def _number(value: str) -> float:
        match = re.search(r"[-+]?\d*\.?\d+", value.replace(",", ""))
        if not match:
            raise ValueError(f"cannot parse numeric GPU field: {value}")
        return float(match.group(0))

    def _query_rows(self) -> list[list[str]]:
        proc = subprocess.run(
            [
                self.executable,
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=max(1.0, self.interval),
            check=False,
        )
        rows = []
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) == 4:
                    rows.append(fields)
        if rows or self.platform != "V5000":
            return rows

        # Older xpu-smi versions only expose the human-readable table.
        proc = subprocess.run(
            [self.executable], capture_output=True, text=True, timeout=max(1.0, self.interval), check=False
        )
        for index, line in enumerate(part for part in proc.stdout.splitlines() if "Default" in part):
            fields = line.split()
            if len(fields) >= 13:
                rows.append([str(index), fields[8], fields[10], fields[12]])
        return rows

    def run(self) -> None:
        if not self.executable:
            return
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["timestamp", "phase", "gpu_index", "memory_used_mb", "memory_total_mb", "utilization_pct"]
            )
            while not self.stop_event.is_set():
                try:
                    now = time.time()
                    for fields in self._query_rows():
                        used, total = self._number(fields[1]), self._number(fields[2])
                        utilization = self._number(fields[3])
                        if total > 0:
                            self.max_memory_pct = max(self.max_memory_pct, 100.0 * used / total)
                        writer.writerow([now, self.tracker.get(), fields[0], used, total, utilization])
                    handle.flush()
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
                self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()


def build_command(
    parameters: Mapping[str, Any], agent_config: Mapping[str, Any], trial_id: int, updates: int
) -> tuple[list[str], Path]:
    verl_root = Path(os.getenv("VERL_ROOT", str(agent_config["verl_root"]))).expanduser().resolve()
    run_parameters = dict(parameters)
    run_parameters.update(
        {
            "trainer.total_training_steps": updates,
            "trainer.total_epochs": max(1, int(parameters.get("trainer.total_epochs", 1))),
            "trainer.experiment_name": f"verl_agent_trial_{trial_id:04d}",
            "trainer.logger": ["console"],
            "trainer.save_freq": -1,
            "trainer.test_freq": -1,
            "trainer.val_before_train": False,
        }
    )
    command = [
        "python3",
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={agent_config.get('config_path', 'config')}",
        f"--config-name={agent_config.get('config_name', 'ppo_megatron_trainer.yaml')}",
        *hydra_overrides(run_parameters),
    ]
    environment_script = os.getenv("VERL_ENV_SCRIPT") or agent_config.get("environment_script")
    if environment_script:
        script = str(Path(str(environment_script)).expanduser().resolve())
        command = ["bash", "-lc", 'source "$1"; shift; exec "$@"', "verl-agent", script, *command]
    return command, verl_root


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_trial(
    parameters: Mapping[str, Any],
    agent_config: Mapping[str, Any],
    trial_id: int,
    stage: str,
    updates: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_root = Path(os.getenv("OUTPUT_PATH", str(agent_config["output_dir"]))).expanduser().resolve()
    trial_dir = output_root / "trials" / f"{trial_id:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "train.log"
    samples_path = trial_dir / "gpu_samples.csv"
    platform = os.getenv("PLATFORM", str(agent_config.get("platform", "V5000")))
    write_json(trial_dir / "parameters.json", dict(parameters))

    command, cwd = build_command(parameters, agent_config, trial_id, updates)
    write_json(trial_dir / "command.json", {"cwd": str(cwd), "argv": command})
    if dry_run:
        return {
            "trial_id": trial_id,
            "stage": stage,
            "platform": platform,
            "dry_run": True,
            "updates_target": updates,
            "parameters": dict(parameters),
            "command": command,
            "cwd": str(cwd),
        }
    if not cwd.exists():
        raise FileNotFoundError(f"verl_root does not exist: {cwd}")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("VERL_LOGGING_LEVEL", "DEBUG")
    tracker = PhaseTracker()
    sampler = GPUSampler(
        samples_path,
        tracker,
        float(agent_config.get("monitor_interval_seconds", 1.0)),
        platform,
    )
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    sampler.start()
    stop_reason = None
    gate_updates = int(agent_config.get("resource_gate_updates", 5))
    hard_limit = float(agent_config.get("resource_memory_limit_pct", 95.0))
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
                tracker.update_from_log(line)
                if FATAL_RE.search(line):
                    stop_reason = "fatal_log"
                    _terminate(process)
                    break
                step_match = STEP_RE.search(line)
                if step_match and int(step_match.group(1)) >= gate_updates and sampler.max_memory_pct >= hard_limit:
                    stop_reason = "resource_gate_memory_limit"
                    _terminate(process)
                    break
        return_code = process.wait()
    finally:
        sampler.stop()
        sampler.join(timeout=5)
        _terminate(process)

    metrics = analyze_trial(
        log_path,
        samples_path,
        warmup_updates=int(agent_config.get("warmup_updates", 5)),
        reward_window=int(agent_config.get("reward_window", 5)),
        reward_thresholds=agent_config.get("reward_thresholds", [0.0, 0.1, 0.2, 0.3]),
    )
    metrics.update(
        {
            "trial_id": trial_id,
            "stage": stage,
            "platform": platform,
            "updates_target": updates,
            "parameters": dict(parameters),
            "return_code": return_code,
            "stop_reason": stop_reason,
            "failure_phase": tracker.get() if stop_reason or return_code != 0 else None,
            "log_path": str(log_path),
            "gpu_samples_path": str(samples_path) if samples_path.exists() else None,
        }
    )
    if metrics["updates_completed"] < updates and not metrics["error"].get("type"):
        metrics["error"] = {
            "type": "INCOMPLETE_TRAINING",
            "evidence": [f"completed {metrics['updates_completed']} of {updates} updates"],
        }
    observed_memory = metrics.get("resource", {}).get("max_observed_memory_pct")
    throughput_limit = float(agent_config.get("throughput_memory_limit_pct", 92.0))
    if stage.startswith("hardware") and observed_memory is not None and observed_memory > throughput_limit:
        metrics["error"] = {
            "type": "MEMORY_HEADROOM_EXCEEDED",
            "evidence": [f"observed phase memory {observed_memory:.2f}% exceeds {throughput_limit:.2f}%"],
        }
        metrics["result"] = "fail"
    if return_code != 0 or stop_reason or metrics["updates_completed"] < updates:
        metrics["result"] = "fail"
    write_json(trial_dir / "trial_report.json", metrics)
    return metrics
