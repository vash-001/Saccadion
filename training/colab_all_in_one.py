"""One-Paste Google Colab Free TPU Launcher for Saccadion-1.

Copy-paste this entire script into a single Google Colab TPU code cell (Runtime -> TPU v2-8/v5e-1),
or execute directly via `%run /content/Saccadion/training/colab_all_in_one.py`.

What it does automatically:
1. Mounts Google Drive for persistent checkpoints, logs, charts, and exported `.saccnet` weights.
2. Clones / updates https://github.com/vash-001/Saccadion.git
3. Installs lightweight dependencies (`optax`, `python-chess`, `zstandard`, `matplotlib`).
4. Downloads & organizes high-Elo chess datasets (Lichess Elite 2500+ & Stockfish Fishtest 3600+).
5. Launches Smart Epoch-by-Epoch TPU Training with full live loss/accuracy/biology telemetry.
6. Displays the 6-panel training dashboard inline and saves `saccadion_best.saccnet` to Google Drive.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ======================== USER CONFIGURATION ========================
REPO_URL = "https://github.com/vash-001/Saccadion.git"
MODEL_TIER = "nano"          # "tiny" (smoke test) | "nano" (fast ablation) | "pro" (full 62M)
DATA_SOURCE = "auto"         # "auto" (Elite + Fishtest) | "lichess_elite" | "fishtest" | "lichess_eval"
MAX_GAMES = 4000             # Increase to 20000-100000 for serious Pro training
EPOCHS = 15                  # Smart 3-stage Neurodevelopmental curriculum across all epochs
BATCH_SIZE = 1024            # Divisible by 8 for TPU v2-8 pmap sharding
USE_GOOGLE_DRIVE = True      # Saves checkpoints & best .saccnet safely across Colab disconnects
# ====================================================================


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


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

    # 3. Install required Python packages
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "optax", "python-chess", "zstandard", "matplotlib"])

    # 4. Import Saccadion training modules
    sys.path.insert(0, str(work_root / "training"))
    import jax
    from saccadion_tpu.config import SaccadionConfig
    from saccadion_tpu.fetch_and_prepare_data import fetch_and_prepare
    from saccadion_tpu.train import train_saccadion_smart

    print("JAX version:", jax.__version__, "| Devices:", jax.devices())

    # 5. Download & organize chess datasets into .sacc shards (cached on Drive)
    data_dir = persist_root / "data"
    shard_dir = data_dir / "shards"
    if not list(shard_dir.glob("train_random_*.sacc")):
        shard_dir = fetch_and_prepare(
            out_dir=data_dir,
            source=DATA_SOURCE,
            max_games=MAX_GAMES,
            val_pct=5,
        )
    else:
        print(f"Using cached organized shards in {shard_dir}")

    # 6. Run Smart Epoch-by-Epoch TPU Training with full telemetry
    cfg = {
        "tiny": SaccadionConfig.tiny,
        "nano": SaccadionConfig.nano,
        "pro": SaccadionConfig.pro,
    }[MODEL_TIER]()

    ckpt_dir = persist_root / f"checkpoints_{MODEL_TIER}"
    export_dir = persist_root / f"exports_{MODEL_TIER}"

    _, history = train_saccadion_smart(
        cfg=cfg,
        shard_dir=shard_dir,
        checkpoint_dir=ckpt_dir,
        export_dir=export_dir,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # 7. Display visual 6-panel dashboard inline in Colab
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
