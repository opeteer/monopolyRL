"""CPU-Only Multi-Core Server Hardware Calibration for Model-3."""

import os
from dataclasses import dataclass
from model_2.system.hardware import _get_cpu_info, _get_mem_info


@dataclass
class HardwareProfileV3:
    cpu_model: str
    physical_cores: int
    logical_threads: int
    total_ram_gb: float
    available_ram_gb: float
    recommended_workers: int
    recommended_envs: int
    recommended_torch_threads: int
    recommended_rollout_steps: int


def detect_hardware() -> HardwareProfileV3:
    """Calibrates worker processes vs PyTorch OpenMP threads strictly for CPU-only execution."""
    cpu_model, physical_cores, logical_threads = _get_cpu_info()
    total_ram_gb, available_ram_gb = _get_mem_info()

    # Reserve ~40% of physical cores for Main PyTorch batched inference & PPO updates,
    # ~60% for parallel single-threaded SharedMemory rollout/search workers.
    rec_torch_threads = max(2, min(16, int(round(physical_cores * 0.45))))
    rec_workers = max(2, min(24, physical_cores - max(1, rec_torch_threads // 2)))
    envs_per_worker = 8 if available_ram_gb >= 12.0 else 4
    rec_envs = rec_workers * envs_per_worker
    rec_rollout = 256 if physical_cores >= 8 else 128

    return HardwareProfileV3(
        cpu_model=cpu_model,
        physical_cores=physical_cores,
        logical_threads=logical_threads,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        recommended_workers=rec_workers,
        recommended_envs=rec_envs,
        recommended_torch_threads=rec_torch_threads,
        recommended_rollout_steps=rec_rollout,
    )


def print_hardware_summary(profile: HardwareProfileV3):
    print("=" * 65)
    print("      MODEL-3 CPU-ONLY SERVER HARDWARE CALIBRATION")
    print("=" * 65)
    print(f"  CPU Model         : {profile.cpu_model}")
    print(f"  Topology          : {profile.physical_cores} Physical Cores / {profile.logical_threads} Threads")
    print(f"  System RAM        : {profile.available_ram_gb:.1f} GB avail / {profile.total_ram_gb:.1f} GB total")
    print(f"  Execution Mode    : CPU-ONLY (RolloutBufferV3 float16 tiles in RAM)")
    print("-" * 65)
    print(f"  Recommended Env Workers     : {profile.recommended_workers} (1-thread each)")
    print(f"  Recommended Parallel Envs   : {profile.recommended_envs}")
    print(f"  Recommended PyTorch Threads : {profile.recommended_torch_threads} (main process)")
    print(f"  Recommended Rollout Horizon : {profile.recommended_rollout_steps}")
    print("=" * 65)
