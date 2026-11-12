"""Hardware Detection and Auto-Configuration for High-Performance RL Training."""

import os
import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class HardwareProfile:
    cpu_model: str
    physical_cores: int
    logical_threads: int
    total_ram_gb: float
    available_ram_gb: float
    device_name: str
    is_cuda: bool
    recommended_workers: int
    recommended_envs: int
    recommended_torch_threads: int
    recommended_rollout_steps: int


def _get_cpu_info():
    """Extracts CPU model, physical cores, and logical threads on Linux without external dependencies."""
    cpu_model = "Generic CPU"
    core_ids = set()
    logical_threads = os.cpu_count() or 4
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line and cpu_model == "Generic CPU":
                    cpu_model = line.split(":", 1)[1].strip()
                elif "core id" in line:
                    core_ids.add(line.split(":", 1)[1].strip())
    except Exception:
        pass

    physical_cores = len(core_ids) if core_ids else max(1, logical_threads // 2)
    return cpu_model, physical_cores, logical_threads


def _get_mem_info():
    """Extracts total and available RAM in GB from /proc/meminfo."""
    total_gb = 8.0
    avail_gb = 6.0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_gb = round(int(line.split()[1]) / (1024 * 1024), 1)
                elif line.startswith("MemAvailable:"):
                    avail_gb = round(int(line.split()[1]) / (1024 * 1024), 1)
    except Exception:
        pass
    return total_gb, avail_gb


def detect_hardware() -> HardwareProfile:
    """Detects CPU topology, RAM, and GPU devices, returning optimal parameters."""
    cpu_model, physical_cores, logical_threads = _get_cpu_info()
    total_ram_gb, available_ram_gb = _get_mem_info()

    is_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if is_cuda else f"CPU ({cpu_model})"

    if is_cuda:
        rec_workers = min(16, max(4, physical_cores))
        rec_torch_threads = min(8, max(2, physical_cores // 2))
    else:
        # On CPU, balance rollout workers and PyTorch BLAS/OpenMP threads
        # For Intel hybrid architecture (10 physical cores / 16 threads):
        # 10 workers for rollout, 6 OpenMP threads for P-core tensor optimization
        rec_workers = min(10, max(2, physical_cores))
        rec_torch_threads = min(6, max(2, physical_cores // 2))

    # 8 envs per worker provides ideal batch amortization
    envs_per_worker = 8
    rec_envs = rec_workers * envs_per_worker
    rec_rollout_steps = 128

    return HardwareProfile(
        cpu_model=cpu_model,
        physical_cores=physical_cores,
        logical_threads=logical_threads,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        device_name=device_name,
        is_cuda=is_cuda,
        recommended_workers=rec_workers,
        recommended_envs=rec_envs,
        recommended_torch_threads=rec_torch_threads,
        recommended_rollout_steps=rec_rollout_steps,
    )


def print_hardware_summary(profile: HardwareProfile):
    """Prints formatted hardware diagnostics and auto-configured allocations."""
    print("=" * 65)
    print("          HARDWARE AUTO-CONFIGURATION PROFILE")
    print("=" * 65)
    print(f" CPU Model:          {profile.cpu_model}")
    print(f" Physical Cores:     {profile.physical_cores} | Logical Threads: {profile.logical_threads}")
    print(f" System Memory:      {profile.available_ram_gb} GB free / {profile.total_ram_gb} GB total")
    print(f" Compute Device:     {profile.device_name}")
    print("-" * 65)
    print(" Recommended Resource Allocation:")
    print(f"  • Environment Workers: {profile.recommended_workers} worker processes (Zero-Copy Shared Memory)")
    print(f"  • Parallel Envs:       {profile.recommended_envs} ({profile.recommended_envs // profile.recommended_workers} envs/worker)")
    print(f"  • PyTorch MKL Threads: {profile.recommended_torch_threads} threads (optimized for P-cores)")
    print(f"  • Rollout Steps:       {profile.recommended_rollout_steps} steps/env ({profile.recommended_envs * profile.recommended_rollout_steps:,} steps/iter)")
    print("=" * 65)
