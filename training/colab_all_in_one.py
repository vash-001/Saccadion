"""One-Paste Google Colab Free TPU Launcher for Saccadion-2 Causal.

Copy-paste into a single Google Colab TPU code cell (Runtime -> TPU v5e-1 / v2-8),
or execute directly via `%run /content/Saccadion/training/colab_all_in_one.py`.

What it does automatically:
1. Mounts Google Drive for persistent checkpoints, logs, charts, and exported `.saccnet` weights.
2. Clones / updates https://github.com/vash-001/Saccadion.git and PURGES cached Python submodules
   so every `git pull` always runs the exact newest code without needing a Colab restart!
3. Optionally compiles (and caches on Drive!) Pure-Rust `saccadion-data` (400x faster pre-decoder)
   and `saccadion` UCI engine (for live Background CPU Stockfish Sparring & Real Elo!).
4. Downloads & organizes Elite chess datasets:
   - Stockfish Fishtest 3600+ Elo evaluated games
   - Lichess Elite 2500+ Elo creative human games
   - Lichess Elite 2000+ Causal Puzzles (16 ground-truth causal concepts!)
5. Pre-decodes shards ONCE into `/content/sacc_cache/` local SSD/RAM (~50x faster epochs: ~25s!).
6. Runs Smart Epoch-by-Epoch Causal TPU Training with a smooth 20 Hz live HTML/SVG cockpit
   and simultaneous Background CPU Stockfish Sparring Arena!
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ======================== USER CONFIGURATION ========================
REPO_URL = "https://github.com/vash-001/Saccadion.git"
MODEL_TIER = "pro"           # "tiny" (smoke test) | "nano" (fast ablation) | "pro" (full 58MB)
DATA_SOURCE = "auto"         # "auto" (75% Fishtest + 25% Lichess 2000+ [%eval] + Diverse Puzzles + Survival)
MAX_GAMES = 24000            # ~2.4M high-Elo positions (18k Fishtest 3600+ + 6k Lichess 2000+ [%eval] + Puzzles + Contrastive!)
MAX_CAUSAL_PUZZLES = 45000   # Balanced across ALL 16 diverse causal/tactical/endgame concepts!
EPOCHS = 20                  # Saccadion-7 Neuro-Apex (Dendritic GF + FlyWire GNN + 3500-PVS + Quiet Masking)
BATCH_SIZE = 512             # Optimal for PRO on TPU v5e-1
LR_MUON = 3.5e-3             # Calibrated Muon peak LR
LR_ADAMW = 5.0e-4            # Calibrated AdamW peak LR
BUILD_RUST_TOOLS = True      # Builds Pure-Rust pre-decoder + UCI engine once & caches on Drive!
ENABLE_CPU_ARENA = True      # Background Stockfish sparring on idle CPU cores -> REAL ELO!
LIVE_HUD_HZ = 20.0           # Smooth 20 Hz flicker-free Colab HTML/SVG cockpit
USE_GOOGLE_DRIVE = True      # Saves checkpoints & best .saccnet safely across Colab disconnects
# ====================================================================


def run_cmd(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    print(">>", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)
    return res.returncode


def _ensure_rust_binaries(work_root: Path, drive_bin_dir: Path) -> None:
    """Restores cached Linux Rust binaries from Drive or builds them once on Colab (~35s)."""
    target_rel = work_root / "target" / "release"
    target_rel.mkdir(parents=True, exist_ok=True)
    bins = ["saccadion-data", "saccadion"]

    # 1. Restore from Google Drive cache if already built on an earlier session!
    if drive_bin_dir.exists() and all((drive_bin_dir / b).exists() for b in bins):
        for b in bins:
            dst = target_rel / b
            shutil.copy2(drive_bin_dir / b, dst)
            os.chmod(dst, 0o755)
        print(f"Restored cached Pure-Rust binaries from {drive_bin_dir}")
        return

    # 2. Ensure cargo is available (minimal rustup install takes ~15s on Colab)
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
    if not Path(cargo).exists():
        print("Installing minimal Rust toolchain for 400x faster pre-decoding & live Rust UCI arena...")
        rc = subprocess.run(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
            shell=True,
            check=False,
        ).returncode
        if rc != 0:
            print("[Warn] Rust install unavailable; using multi-process Python FastCache fallback.")
            return

    env = os.environ.copy()
    env["PATH"] = f"{Path.home() / '.cargo' / 'bin'}:{env.get('PATH', '')}"
    print("Building Pure-Rust `saccadion-data` + `saccadion` UCI in --release (cached on Drive)...")
    rc = subprocess.run(
        [cargo, "build", "--release", "-p", "saccadion-data", "-p", "saccadion-uci"],
        cwd=str(work_root),
        env=env,
        check=False,
    ).returncode
    if rc == 0:
        drive_bin_dir.mkdir(parents=True, exist_ok=True)
        for b in bins:
            src = target_rel / b
            if src.exists():
                shutil.copy2(src, drive_bin_dir / b)
        print(f"Cached Pure-Rust binaries -> {drive_bin_dir}")
    else:
        print("[Warn] Rust build skipped; falling back cleanly to multi-process Python FastCache.")


def main() -> None:
    in_colab = "google.colab" in sys.modules or Path("/content").exists()
    work_root = Path("/content/Saccadion") if in_colab else Path.cwd()

    # 1. Mount Google Drive if inside Colab
    if in_colab and USE_GOOGLE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        persist_root = Path("/content/drive/MyDrive/Saccadion_TPU")
    else:
        persist_root = work_root / "tpu_runs"
    persist_root.mkdir(parents=True, exist_ok=True)

    # 2. Clone or update Saccadion from GitHub
    if in_colab:
        if not work_root.exists():
            run_cmd(["git", "clone", REPO_URL, str(work_root)])
        else:
            run_cmd(["git", "pull", "--ff-only"], cwd=work_root)

    # 3. Install lightweight Python packages
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "optax", "python-chess", "zstandard", "matplotlib"])

    # 4. Build or restore Pure-Rust tools (v12 = Saccadion-7 Neuro-Apex: Dendritic GF + 3500-PVS!)
    if in_colab and BUILD_RUST_TOOLS:
        _ensure_rust_binaries(work_root, persist_root / "bin_v12")

    # 5. Purge any stale in-memory `saccadion_tpu.*` submodules before importing!
    train_pkg_root = str(work_root / "training")
    if train_pkg_root not in sys.path:
        sys.path.insert(0, train_pkg_root)
    for mod_name in list(sys.modules.keys()):
        if mod_name == "saccadion_tpu" or mod_name.startswith("saccadion_tpu."):
            del sys.modules[mod_name]

    import jax
    from saccadion_tpu.config import SaccadionConfig
    from saccadion_tpu.fetch_and_prepare_data import (
        download_and_pack_elite_causal_puzzles,
        fetch_and_prepare,
        generate_blunder_brilliant_shard,
        generate_looming_blunder_shard,
    )
    from saccadion_tpu.train import train_saccadion_smart

    print("JAX version:", jax.__version__, "| Devices:", jax.devices())

    # 6. Download & organize compact diverse chess dataset (92/8 clean evals + 16-theme balanced puzzles + survival)
    data_dir = persist_root / "data_v7_apex"
    shard_dir = data_dir / "shards"
    if not list(shard_dir.glob("train_random_*.sacc")):
        shard_dir = fetch_and_prepare(
            out_dir=data_dir,
            source=DATA_SOURCE,
            max_games=MAX_GAMES,
            val_pct=5,
            include_causal_puzzles=True,
            max_puzzles=MAX_CAUSAL_PUZZLES,
        )
    else:
        print(f"Using cached organized shards in {shard_dir}")
        if MAX_CAUSAL_PUZZLES > 0 and not (shard_dir / "train_random_puzzles.sacc").exists():
            download_and_pack_elite_causal_puzzles(
                shard_dir, max_puzzles=MAX_CAUSAL_PUZZLES, val_pct=5
            )
        if not (shard_dir / "train_random_looming.sacc").exists():
            generate_looming_blunder_shard(shard_dir, max_samples=40000)
        if not (shard_dir / "train_random_contrastive.sacc").exists():
            generate_blunder_brilliant_shard(shard_dir, max_samples=60000)

    # 7. Launch Smart Epoch-by-Epoch Causal TPU Training with 50x FastCache, 20Hz HUD, & CPU Arena
    cfg = {
        "tiny": SaccadionConfig.tiny,
        "nano": SaccadionConfig.nano,
        "pro": SaccadionConfig.pro,
    }[MODEL_TIER]()

    ckpt_dir = persist_root / f"checkpoints_{MODEL_TIER}_v7"
    export_dir = persist_root / f"exports_{MODEL_TIER}_v7"
    fast_cache = Path("/content/sacc_cache") if in_colab else (data_dir / "cache")

    _, history = train_saccadion_smart(
        cfg=cfg,
        shard_dir=shard_dir,
        checkpoint_dir=ckpt_dir,
        export_dir=export_dir,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr_muon=LR_MUON,
        lr_adamw=LR_ADAMW,
        fast_cache_dir=fast_cache,
        enable_cpu_arena=ENABLE_CPU_ARENA,
        live_hud_hz=LIVE_HUD_HZ,
    )

    dash_png = ckpt_dir / "training_dashboard.png"
    if in_colab and dash_png.exists():
        from IPython.display import Image, display
        display(Image(filename=str(dash_png)))

    print("\nFinished! Download your Pure-Rust weights from:")
    print(" ", export_dir / "saccadion_best.saccnet")
    print("Metrics CSV :", ckpt_dir / "metrics_history.csv")
    print("Dashboard   :", dash_png)


if __name__ == "__main__":
    main()
