"""Session Management & 24/7 Crash-Safe Resume Support for Model-3."""

import os
import glob
from typing import Dict, Any, Optional

from model_2.system.session_manager import SessionManager


def resolve_latest_checkpoint(resume_arg: Optional[str], base_dir: str = "runs_model_3") -> Optional[str]:
    """Resolves '--resume latest' or explicit checkpoint path cleanly across sessions."""
    if not resume_arg or resume_arg.lower() == "none":
        return None
    if resume_arg.lower() != "latest":
        return resume_arg if os.path.exists(resume_arg) else None

    if not os.path.exists(base_dir):
        return None

    candidates = []
    for root, _, files in os.walk(base_dir):
        for fname in ("checkpoint_latest.pt", "interrupted_checkpoint.pt"):
            if fname in files:
                fpath = os.path.join(root, fname)
                candidates.append((os.path.getmtime(fpath), fpath))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


class SessionManagerV3(SessionManager):
    """Extends Model-2 SessionManager for Model-3 24/7 atomic checkpointing & resume."""

    def __init__(
        self,
        session_name: Optional[str] = None,
        base_dir: str = "runs_model_3",
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(session_name=session_name, base_dir=base_dir, config_dict=config_dict)

    @property
    def latest_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")

    def find_latest_checkpoint(self) -> Optional[str]:
        if os.path.exists(self.latest_checkpoint_path):
            return self.latest_checkpoint_path
        interrupted = os.path.join(self.checkpoint_dir, "interrupted_checkpoint.pt")
        if os.path.exists(interrupted):
            return interrupted
        return super().find_latest_checkpoint()
