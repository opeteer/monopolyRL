"""High-Performance System and Training Management Utilities."""

from model_2.system.hardware import detect_hardware, print_hardware_summary, HardwareProfile
from model_2.system.config_presets import TrainingConfig, PRESETS, get_preset, list_presets
from model_2.system.session_manager import SessionManager

__all__ = [
    "detect_hardware",
    "print_hardware_summary",
    "HardwareProfile",
    "TrainingConfig",
    "PRESETS",
    "get_preset",
    "list_presets",
    "SessionManager",
]
