"""Session Management, Metrics Logging, and Interruption Handling for RL Training."""

import os
import sys
import time
import json
import signal
import glob
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch


class SessionManager:
    """Manages individual training sessions, telemetry logging, and checkpointing."""

    def __init__(
        self,
        session_name: Optional[str] = None,
        base_dir: str = "sessions",
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"run_{timestamp}"
        self.session_dir = os.path.join(base_dir, self.session_name)
        self.checkpoint_dir = os.path.join(self.session_dir, "checkpoints")
        self.logs_dir = os.path.join(self.session_dir, "logs")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.metrics_file = os.path.join(self.logs_dir, "metrics.jsonl")
        self.summary_file = os.path.join(self.session_dir, "session_summary.json")
        self.config_file = os.path.join(self.session_dir, "session_config.json")

        self.config_dict = config_dict or {}
        with open(self.config_file, "w") as f:
            json.dump(self.config_dict, f, indent=2)

        self.start_time = time.time()
        self.interrupted = False
        self._prev_sigint = None
        self._prev_sigterm = None
        self._register_signals()

    def _register_signals(self):
        """Catches SIGINT (Ctrl+C) and SIGTERM to save checkpoints before exiting."""
        def handler(sig, frame):
            print("\n\n[SessionManager] Caught interruption signal! Preparing safe checkpoint...")
            self.interrupted = True

        self._prev_sigint = signal.signal(signal.SIGINT, handler)
        self._prev_sigterm = signal.signal(signal.SIGTERM, handler)

    def restore_signals(self):
        """Restores original signal handlers."""
        if self._prev_sigint:
            signal.signal(signal.SIGINT, self._prev_sigint)
        if self._prev_sigterm:
            signal.signal(signal.SIGTERM, self._prev_sigterm)

    def log_iteration(self, metrics: Dict[str, Any]):
        """Appends iteration telemetry to JSON Lines metrics log."""
        metrics_record = dict(metrics)
        metrics_record["timestamp"] = time.time()
        metrics_record["iso_time"] = datetime.now().isoformat()
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics_record) + "\n")

    def find_latest_checkpoint(self) -> Optional[str]:
        """Finds the latest .pt checkpoint in the session directory."""
        ckpts = glob.glob(os.path.join(self.checkpoint_dir, "main_iter_*.pt"))
        if not ckpts:
            interrupted = os.path.join(self.checkpoint_dir, "interrupted_checkpoint.pt")
            if os.path.exists(interrupted):
                return interrupted
            return None
        # Sort by iteration number in filename
        ckpts.sort(key=lambda x: int(os.path.basename(x).split("_")[2].split(".")[0]) if "iter" in x else 0)
        return ckpts[-1]

    def save_summary(self, summary_data: Dict[str, Any]):
        """Saves final session summary report."""
        elapsed = time.time() - self.start_time
        summary = {
            "session_name": self.session_name,
            "session_dir": self.session_dir,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_minutes": round(elapsed / 60.0, 2),
            "interrupted": self.interrupted,
            **summary_data,
        }
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        return self.summary_file
