"""High-Performance System and Training Management Utilities."""

from monopoly_marl.system.hardware import detect_hardware, print_hardware_summary, HardwareProfile
from monopoly_marl.system.config_presets import TrainingConfig, PRESETS, get_preset, list_presets
from monopoly_marl.system.session_manager import SessionManager

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
