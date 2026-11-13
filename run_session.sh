#!/usr/bin/env bash
# ==============================================================================
# Monopoly RL - High Performance Tournament Training Session Runner
# ==============================================================================
# Utilizes zero-copy Shared Memory vectorized multiprocessing across physical CPU cores
# for ultra-fast training iterations without quality degradation.
# ==============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Locate Virtualenv Python
if [ -f "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python"
elif [ -f "/home/opeteer/cardGameRL/.venv/bin/python" ]; then
    PYTHON_BIN="/home/opeteer/cardGameRL/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

show_help() {
    echo "======================================================================"
    echo "             Monopoly RL Multi-Agent Training Launcher                "
    echo "======================================================================"
    echo "Usage: ./run_session.sh [OPTIONS]"
    echo ""
    echo "Available Presets:"
    echo "  --smoke        : Quick 10k steps smoke test (<3s)"
    echo "  --fast         : Fast 250k steps prototyping session (~30-45s)"
    echo "  --standard     : Standard 1M steps tournament training (~2-3 min)"
    echo "  --tournament   : Full 5M steps tournament competitive training (~10-15 min)"
    echo "  --overnight    : Deep 10M steps tournament master training (~25-35 min)"
    echo "  --deep         : Grandmaster 25M steps deep league training (~1-1.5 hr)
  --grandmaster  : Ultra-deep 50M steps Grandmaster league training (~2.5 hr)"
    echo ""
    echo "Operations:"
    echo "  --auto         : Auto-detect CPU & RAM, choosing optimal workers & envs"
    echo "  --resume       : Resume from latest session checkpoint"
    echo "  --benchmark    : Benchmark FPS across worker counts (finds max speed)"
    echo "  --name <NAME>  : Set custom session name (saved to sessions/<NAME>/)"
    echo "  --help         : Show this help message"
    echo "======================================================================"
}

run_benchmark() {
    echo "--> Running hardware throughput benchmark..."
    $PYTHON_BIN -c "
from monopoly_marl.system.hardware import detect_hardware, print_hardware_summary
from monopoly_marl.engine.vec_env import SharedMemoryMonopolyVecEnv
import time

hw = detect_hardware()
print_hardware_summary(hw)

for w in [4, 8, hw.recommended_workers]:
    n_envs = w * 8
    configs = [dict(max_rounds=120, seed=None, allow_trade=True) for i in range(n_envs)]
    vec = SharedMemoryMonopolyVecEnv(configs, num_workers=w)
    vec.reset()
    start = time.time()
    steps = 150
    for _ in range(steps):
        vec.step()
    el = time.time() - start
    vec.close()
    tot = steps * n_envs
    print(f'[*] Workers: {w:2d} | Envs: {n_envs:3d} | Steps: {tot:,} in {el:.2f}s -> {int(tot/el):,} steps/sec')
"
}

# If no arguments provided, show interactive selector
if [ $# -eq 0 ]; then
    show_help
    echo ""
    read -p "Select preset [1: smoke, 2: fast, 3: standard, 4: tournament, 5: overnight, b: benchmark]: " choice
    case "$choice" in
        1) PRESET="smoke_test" ;;
        2) PRESET="fast_dev" ;;
        3) PRESET="standard_1m" ;;
        4) PRESET="tournament_5m" ;;
        5) PRESET="overnight_10m" ;;
        b|B) run_benchmark; exit 0 ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
    $PYTHON_BIN main_train.py --preset "$PRESET"
    exit 0
fi

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            ARGS+=("--preset" "smoke_test")
            shift ;;
        --fast)
            ARGS+=("--preset" "fast_dev")
            shift ;;
        --standard)
            ARGS+=("--preset" "standard_1m")
            shift ;;
        --tournament)
            ARGS+=("--preset" "tournament_5m")
            shift ;;
        --overnight)
            ARGS+=("--preset" "overnight_10m")
            shift ;;
        --deep)
            ARGS+=("--preset" "deep_league_25m")
            shift ;;
        --grandmaster|--50m)
            ARGS+=("--preset" "grandmaster_50m")
            shift ;;
        --auto)
            ARGS+=("--auto_hardware")
            shift ;;
        --resume)
            ARGS+=("--resume" "latest")
            shift ;;
        --benchmark)
            run_benchmark
            exit 0 ;;
        --name)
            ARGS+=("--session_name" "$2")
            shift 2 ;;
        --help|-h)
            show_help
            exit 0 ;;
        *)
            ARGS+=("$1")
            shift ;;
    esac
done

exec $PYTHON_BIN main_train.py "${ARGS[@]}"
