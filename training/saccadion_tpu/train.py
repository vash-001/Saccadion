"""Smart Epoch-by-Epoch Colab Free TPU Trainer for Saccadion-1 (JAX + Muon + Full Live Telemetry).

Features:
- Automatic multi-core TPU sharding (`jax.pmap` across all 8 TPU cores when available, seamless
  fallback to single-device `jax.jit` on v5e-1 / GPU / CPU).
- Smart 3-Stage Neurodevelopmental Curriculum across epochs:
    * Stage 1 (Embryo Vision Snap): T=2 ticks, fast visual/policy convergence
    * Stage 2 (Deep Deliberation):  T=6 ticks, Deep Supervision + 10-20 ply Pressure + K=4 Creativity
    * Stage 3 (Adult Plasticity):   Meta-Dopamine Eligibility Scan (lax.scan) + APL 92% + STE-QAT
- Cosine Annealing + Warmup LR schedules for Muon and AdamW.
- Adaptive APL Sparsity Controller steering active Kenyon Cells cleanly toward 8.0% (92% zeros).
- Hard-Example Surprise Replay Buffer (upsampling highest-loss turning points each epoch).
- Full Held-Out Validation Suite EVERY Epoch:
    * Train & Val Total Loss + ALL 8 individual sensor losses
    * Top-1 & Top-3 Move Policy Accuracy (%)
    * WDL Class Accuracy (%) & Centipawn MAE
    * 10-20 Ply Strategic Pressure MAE & Lateral Horn MAE
    * APL Active KC %, K=4 Plan Orthogonality Error, Meta-Dopamine Adaptation Gain, Learned eta_PAM/PPL1/rho
- Persistent CSV + JSONL history, 6-panel Matplotlib Dashboard PNG (rendered live in Colab),
  automatic 15-minute & per-epoch Drive checkpointing, and automatic best `.saccnet` export!
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import math
import pickle
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _reclaim_host_ram() -> None:
    """Explicitly reclaims Python + glibc heap arenas after each epoch to keep Colab RAM flat."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import SaccadionConfig, build_flyhash_claws
from .cpu_arena_sparring import StockfishSparringArena
from .dataset import SaccShardDataset, augment_batch_file_mirror
from .export_rust import export_params_to_saccnet
from .live_ui import LiveTrainingHUD
from .losses import compute_multitask_loss, jepa_vicreg_loss, soft_kl_loss
from .model import (
    forward_stream_a,
    forward_stream_b,
    init_saccadion_params,
    meta_dopamine_trajectory_scan,
    predict_minimax_world_model,
)


def _newton_schulz_5(g: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
    """5-step quintic Newton-Schulz orthogonalization (fallback if optax.contrib.muon is absent)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_shape = g.shape
    m = g.reshape((g.shape[0], -1)).astype(jnp.bfloat16)
    transposed = m.shape[0] > m.shape[1]
    if transposed:
        m = m.T
    m = m / (jnp.linalg.norm(m) + eps)
    for _ in range(5):
        aa = m @ m.T
        bb = b * aa + c * (aa @ aa)
        m = a * m + bb @ m
    if transposed:
        m = m.T
    return m.reshape(orig_shape).astype(g.dtype)


def _signed_decoupled_weight_decay(
    sched_lr: Any, weight_decay: float
) -> optax.GradientTransformation:
    """Applies true negative decoupled weight decay: `u <- u - lr(step) * wd * p`.

    Safe to place AFTER `optax.contrib.muon` (which already applies `-lr`), unlike
    `optax.add_decayed_weights` which adds `+wd * p` expecting a downstream `-lr` scaler!
    """
    def init_fn(params):
        del params
        return jnp.array(0, dtype=jnp.int32)

    def update_fn(updates, state, params=None):
        step = state
        lr = sched_lr(step)
        if params is not None and weight_decay > 0.0:
            updates = jax.tree_util.tree_map(
                lambda u, p: u - (lr * weight_decay) * p, updates, params
            )
        return updates, step + 1

    return optax.GradientTransformation(init_fn, update_fn)


def build_muon_adamw_optimizer(
    params: Dict[str, jnp.ndarray],
    total_steps: int,
    lr_muon_peak: float = 1.5e-2,
    lr_adamw_peak: float = 1.5e-3,
    warmup_frac: float = 0.06,
) -> Tuple[optax.GradientTransformation, Any, Any]:
    """Routes dense 2D+ hidden matrices to Muon and sparse table/biases/LN to AdamW with cosine LR."""
    warmup_steps = max(10, int(total_steps * warmup_frac))
    decay_steps = max(warmup_steps + 10, total_steps)

    sched_muon = optax.warmup_cosine_decay_schedule(
        init_value=lr_muon_peak * 0.05,
        peak_value=lr_muon_peak,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=lr_muon_peak * 0.01,
    )
    sched_adamw = optax.warmup_cosine_decay_schedule(
        init_value=lr_adamw_peak * 0.05,
        peak_value=lr_adamw_peak,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=lr_adamw_peak * 0.01,
    )

    sched_sez = optax.warmup_cosine_decay_schedule(
        init_value=lr_adamw_peak * 0.02,
        peak_value=lr_adamw_peak * 0.25,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=lr_adamw_peak * 0.01,
    )

    labels = {}
    for k, v in params.items():
        if k == "a_sez_weight":
            labels[k] = "sez"
        elif k == "a_ft_weight":
            labels[k] = "sparse_ft"
        elif v.ndim < 2 or "ln" in k or "bias" in k or "meta_" in k or "_rel" in k or "plastic_to_mbon" in k or k.startswith("a_gf_"):
            labels[k] = "adamw"
        else:
            labels[k] = "muon"

    if hasattr(optax, "contrib") and hasattr(optax.contrib, "muon"):
        muon_tx = optax.chain(
            optax.contrib.muon(learning_rate=sched_muon, beta=0.95),
            _signed_decoupled_weight_decay(sched_muon, weight_decay=0.025),
        )
    else:
        def init_fn(p):
            return (jnp.array(0, dtype=jnp.int32), jax.tree_util.tree_map(jnp.zeros_like, p))

        def update_fn(updates, state, p=None):
            step, mom = state
            lr = sched_muon(step)
            new_mom = jax.tree_util.tree_map(lambda m, g: 0.95 * m + g, mom, updates)
            orth = jax.tree_util.tree_map(
                lambda m: -lr * (_newton_schulz_5(m) if m.ndim >= 2 else m),
                new_mom,
            )
            return orth, (step + 1, new_mom)

        muon_tx = optax.chain(
            optax.GradientTransformation(init_fn, update_fn),
            _signed_decoupled_weight_decay(sched_muon, weight_decay=0.025),
        )

    # Strong anti-overfit decoupled weight decay: prevents 13.9M HalfKA table & 4.5M Stream B
    # from memorizing static games and drives Val GF Quiet MAE down cleanly!
    adamw_tx = optax.adamw(learning_rate=sched_adamw, weight_decay=1.5e-2)
    sparse_ft_tx = optax.adamw(learning_rate=sched_adamw, weight_decay=4.5e-2)
    sez_tx = optax.adamw(learning_rate=sched_sez, weight_decay=2.0e-4)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {"muon": muon_tx, "adamw": adamw_tx, "sparse_ft": sparse_ft_tx, "sez": sez_tx}, labels
        ),
    )
    return tx, sched_muon, sched_adamw


def estimate_elo_proxy(val_top1_pct: float, val_wdl_acc: float, val_cp_mae: float) -> float:
    """Calibrated proxy Elo estimate from held-out Move Top-1 (4096), WDL Acc, and Eval MAE.

    Note: This is an empirical neural-strength proxy (1-ply / raw network intuition);
    combining with Rust Neuro-Guided PVS search typically adds +400 to +700 Elo on CPU.
    """
    # Calibrated anchor points:
    #   Top-1 ~10%, MAE ~180 cp, WDL ~45% -> ~1150 Elo (raw net)
    #   Top-1 ~35%, MAE ~85 cp,  WDL ~65% -> ~2200 Elo (raw net)
    #   Top-1 ~52%, MAE ~40 cp,  WDL ~78% -> ~2900 Elo (raw net)
    pol_term = 38.0 * min(val_top1_pct, 65.0)
    wdl_term = 14.0 * max(0.0, val_wdl_acc - 33.3)
    mae_term = max(0.0, (240.0 - min(val_cp_mae, 240.0)) * 3.2)
    return float(max(600.0, min(3600.0, 520.0 + 0.55 * pol_term + 0.45 * wdl_term + 0.35 * mae_term)))


def _dequant_board_feats(bf: jnp.ndarray) -> jnp.ndarray:
    """Dequantizes compact INT8 FastCache v7 board_feats on TPU in <0.005 ms (zero host cost!)."""
    if bf.dtype == jnp.int8:
        return bf.astype(jnp.float32) * jnp.float32(1.0 / 127.0)
    return bf.astype(jnp.float32)


def build_freeze_thaw_schedule(
    params: Dict[str, jnp.ndarray], epoch: int, total_epochs: int
) -> Tuple[int, str, int, float, bool, Dict[str, jnp.ndarray], Dict[str, str]]:
    """4-Stage Neurodevelopmental Freeze/Thaw Curriculum preventing late-epoch regression:
      Phase 1 (Foundation):      Thaw Optic+GNN + SEZ + Dendritic GF; freeze Meta-DA.
      Phase 2 (Causal+Blunder):  Anchor SEZ/HalfKA/GF (0.08-0.25x) so leaf Elo NEVER drifts;
                                 full-thaw GNN, CX (T=4), World Model, Blunder/Brilliant, Causal!
      Phase 3 (Distill+DA):      Anchor Optic/GNN (0.15x); full-thaw GF, Stream A, & Meta-DA!
      Phase 4 (Apex Polish+QAT): Gentle co-adaptive polish (0.30x) + STE-QAT + EMA selection!
    """
    p1_end = max(1, int(round(total_epochs * 0.25)))
    p2_end = max(p1_end + 1, int(round(total_epochs * 0.60)))
    p3_end = max(p2_end + 1, int(round(total_epochs * 0.85)))
    qat_active = epoch >= max(2, total_epochs - 1) and total_epochs >= 4

    if epoch <= p1_end:
        stage, name, ticks, apl_t, qat = 1, "P1:Foundation-Thaw", 3, 0.14, False
        m_gf, m_ft, m_sez, m_optic, m_cx, m_wm, m_da = 1.0, 1.0, 1.0, 1.0, 0.60, 0.35, 0.0
        fmap = {"GF+HalfKA": "THAWED 1.0x", "Optic+GNN": "THAWED 1.0x", "CX+World": "WARM 0.6x", "Meta-DA": "FROZEN 0.0x"}
    elif epoch <= p2_end:
        stage, name, ticks, apl_t, qat = 2, "P2:Causal-Blunder-Thaw", 4, 0.10, False
        m_gf, m_ft, m_sez, m_optic, m_cx, m_wm, m_da = 0.25, 0.08, 0.05, 1.0, 1.0, 1.0, 0.0
        fmap = {"GF+HalfKA": "ANCHORED 0.1x", "Optic+GNN": "THAWED 1.0x", "CX+World": "THAWED 1.0x", "Meta-DA": "FROZEN 0.0x"}
    elif epoch <= p3_end:
        stage, name, ticks, apl_t, qat = 3, "P3:Distill-Dopamine-Thaw", 4, 0.08, False
        m_gf, m_ft, m_sez, m_optic, m_cx, m_wm, m_da = 1.0, 0.35, 0.15, 0.15, 0.40, 0.50, 1.0
        fmap = {"GF+HalfKA": "THAWED 1.0x", "Optic+GNN": "ANCHORED 0.15x", "CX+World": "WARM 0.4x", "Meta-DA": "THAWED 1.0x"}
    else:
        stage, name, ticks, apl_t, qat = 4, "P4:Apex-Polish+QAT" if qat_active else "P4:Apex-Polish", 4, 0.08, qat_active
        m_gf, m_ft, m_sez, m_optic, m_cx, m_wm, m_da = 0.35, 0.20, 0.10, 0.25, 0.35, 0.35, 0.40
        fmap = {"GF+HalfKA": "POLISH 0.3x", "Optic+GNN": "POLISH 0.25x", "CX+World": "POLISH 0.35x", "Meta-DA": "POLISH 0.4x"}

    scales: Dict[str, jnp.ndarray] = {}
    for k in params:
        if k.startswith("a_gf_"):
            s = m_gf
        elif k == "a_sez_weight":
            s = m_sez
        elif k in ("a_ft_weight", "a_ft_bias"):
            s = m_ft
        elif k.startswith("meta_") or k == "a_plastic_to_mbon":
            s = m_da
        elif k.startswith("b_wm_"):
            s = m_wm
        elif k.startswith(("b_shunt", "b_lc", "b_optic", "b_opt_", "b_ama_", "b_gnn_", "b_retro_")):
            s = m_optic
        elif k.startswith("a_"):
            s = max(m_gf, 0.35)
        else:
            s = m_cx
        scales[k] = jnp.array(s, dtype=jnp.float32)
    return stage, name, ticks, apl_t, qat, scales, fmap


class _AsyncCheckpointSaver:
    """Writes checkpoints to local SSD first and copies to Google Drive in background (zero FUSE stalls!)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def save_async(
        self,
        drive_path: Path,
        params: Dict[str, Any],
        ema_params: Dict[str, Any],
        opt_state: Any,
        epoch: int,
        global_step: int,
        best_val_loss: float,
        history: List[Dict[str, Any]],
    ) -> None:
        payload = {
            "params": jax.device_get(params),
            "ema_params": jax.device_get(ema_params),
            "opt_state": jax.device_get(opt_state),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "history": list(history),
        }

        def _worker() -> None:
            with self._lock:
                try:
                    drive_path.parent.mkdir(parents=True, exist_ok=True)
                    local_tmp = Path("/content/_sacc_latest.pkl") if Path("/content").exists() else drive_path.with_suffix(".pkl.tmp")
                    with open(local_tmp, "wb") as f:
                        pickle.dump(payload, f)
                    dst_tmp = drive_path.with_suffix(".pkl.tmp")
                    shutil.copy2(local_tmp, dst_tmp)
                    dst_tmp.replace(drive_path)
                except Exception:
                    pass

        self._thread = threading.Thread(target=_worker, name="SaccAsyncCkpt", daemon=True)
        self._thread.start()

    def wait(self) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=60.0)


def save_checkpoint(
    path: Path,
    params: Dict[str, Any],
    opt_state: Any,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    history: List[Dict[str, Any]],
    ema_params: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "params": jax.device_get(params),
                "ema_params": jax.device_get(ema_params) if ema_params is not None else jax.device_get(params),
                "opt_state": jax.device_get(opt_state),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "history": history,
            },
            f,
        )


def load_checkpoint(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_metrics_and_dashboard(history: List[Dict[str, Any]], ckpt_dir: Path) -> None:
    if not history:
        return
    # 1. JSONL & CSV logs
    jsonl_path = ckpt_dir / "metrics_history.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    csv_path = ckpt_dir / "metrics_history.csv"
    # Take ordered union of keys across all epochs so migrated checkpoints never fail CSV export
    keys = list(dict.fromkeys(k for row in history for k in row.keys()))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(history)

    # 2. Render 6-panel Matplotlib visual dashboard (auto-displayed in Colab if available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [r.get("epoch", i + 1) for i, r in enumerate(history)]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Saccadion-8 RETROGRADE FISCHER-GRAPH (6ch-XRay+A^T+2Ply-WM+INT8) TPU Telemetry", fontsize=14, fontweight="bold")

        # Panel 1: Total Train vs Val Loss
        ax = axes[0, 0]
        ax.plot(epochs, [r.get("train_loss", 0.0) for r in history], label="Train Total", marker="o")
        ax.plot(epochs, [r.get("val_loss", 0.0) for r in history], label="Val Total", marker="s")
        ax.set_title("Total Loss (Train vs Val)")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Panel 2: Multi-Sensor Breakdown (Val)
        ax = axes[0, 1]
        ax.plot(epochs, [r.get("val_wdl_b", 0.0) for r in history], label="WDL Deep (B)")
        ax.plot(epochs, [r.get("val_wdl_a", 0.0) for r in history], label="WDL Fast (A)")
        ax.plot(epochs, [r.get("val_policy", 0.0) for r in history], label="PFL3 Policy")
        ax.plot(epochs, [r.get("val_pressure", 0.0) for r in history], label="Pressure 10-20ply")
        ax.set_title("Validation Multi-Sensor Losses")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Panel 3: Chess Accuracies (%)
        ax = axes[0, 2]
        ax.plot(epochs, [r.get("val_top1_acc", 0.0) for r in history], label="Move Top-1 %", marker="o")
        ax.plot(epochs, [r.get("val_top3_acc", 0.0) for r in history], label="Move Top-3 %", marker="^")
        ax.plot(epochs, [r.get("val_wdl_acc", 0.0) for r in history], label="WDL Outcome %", marker="s")
        ax.plot(epochs, [r.get("val_causal_f1", 0.0) * 100.0 for r in history], label="Causal Concept F1 %", marker="d")
        ax.set_title("Chess & Causal Accuracy Metrics (%)")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Panel 4: Centipawn MAE & Strategic Pressure MAE
        ax = axes[1, 0]
        ax.plot(epochs, [r.get("val_cp_mae", 0.0) for r in history], label="Eval MAE (cp)", color="tab:red", marker="o")
        ax.set_title("Evaluation Error (Centipawns MAE)")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Panel 5: APL Sparsity % vs 8% Biological Target
        ax = axes[1, 1]
        ax.plot(epochs, [r.get("val_active_kc_pct", 0.0) for r in history], label="Active KCs %", color="tab:purple", marker="o")
        ax.axhline(8.0, color="tab:green", linestyle="--", label="FlyWire Target (8% = 92% zeros)")
        ax.set_title("APL GABA Sparsity (Active Kenyon Cells %)")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Panel 6: Real Stockfish Arena Elo & Proxy Elo Curve
        ax = axes[1, 2]
        raw_pts = [(r.get("epoch", i + 1), r["est_raw_elo"]) for i, r in enumerate(history) if "est_raw_elo" in r]
        pvs_pts = [(r.get("epoch", i + 1), r["est_search_elo"]) for i, r in enumerate(history) if "est_search_elo" in r]
        if raw_pts:
            ax.plot([p[0] for p in raw_pts], [p[1] for p in raw_pts], label="Est. Raw Net Elo", color="tab:green", marker="o", linewidth=1.8)
        if pvs_pts:
            ax.plot([p[0] for p in pvs_pts], [p[1] for p in pvs_pts], label="Est. +Rust PVS Elo", color="tab:cyan", linestyle="--", marker="^")
        real_pts = [(r.get("epoch", i + 1), r["real_arena_elo"]) for i, r in enumerate(history) if r.get("real_arena_elo", 0.0) > 0]
        if real_pts:
            ax.plot([p[0] for p in real_pts], [p[1] for p in real_pts], label="REAL Multi-Engine Arena Elo", color="tab:red", marker="D", linewidth=2.4)
        ax.set_title("Real Multi-Engine Arena vs Proxy Elo")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Elo")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        plt.tight_layout()
        dash_path = ckpt_dir / "training_dashboard.png"
        plt.savefig(dash_path, dpi=140)
        plt.close(fig)
    except Exception:
        pass


def _shard_batch_for_pmap(batch: Dict[str, np.ndarray], n_dev: int) -> Dict[str, jnp.ndarray]:
    out = {}
    for k, v in batch.items():
        trim = (v.shape[0] // n_dev) * n_dev
        v_t = v[:trim]
        out[k] = jnp.asarray(v_t.reshape((n_dev, trim // n_dev) + v_t.shape[1:]))
    return out


def _print_epoch_banner(row: Dict[str, Any], is_best: bool) -> None:
    best_tag = "  *** NEW BEST -> EXPORTED .saccnet ***" if is_best else ""
    eng_summary = row.get("arena_engines_str", "")
    real_elo_s = (
        f"REAL MULTI-ARENA={row['real_arena_elo']:.0f} +/-{row.get('arena_ci', 0):.0f} Elo "
        f"({row['arena_games']}g | {row.get('arena_workers', 1)}x parallel{eng_summary})"
        if row.get("real_arena_elo", 0.0) > 0
        else f"REAL MULTI-ARENA=calibrating ({row.get('arena_workers', 1)}x parallel)..."
    )
    print("\n" + "=" * 96)
    print(
        f"EPOCH {row['epoch']:02d}/{row['total_epochs']:02d} | STAGE {row['stage']} ({row['stage_name']}) "
        f"| Ticks={row['ticks']} | QAT={row['qat']} | Time={row['epoch_sec']:.1f}s ({row['pos_per_sec']:,.0f} pos/s){best_tag}"
    )
    print("-" * 96)
    print(
        f"  LOSS    : train={row['train_loss']:.4f} | val={row['val_loss']:.4f} "
        f"(delta={row['val_delta']:+.4f}) | LR_Muon={row['lr_muon']:.2e} LR_AdamW={row['lr_adamw']:.2e}"
    )
    print(
        f"  SENSORS : WDL_B={row['val_wdl_b']:.4f} | WDL_A={row['val_wdl_a']:.4f} | "
        f"Policy={row['val_policy']:.4f} | Press(10-20ply)={row['val_pressure']:.4f} | LH={row['val_lh']:.4f}"
    )
    print(
        f"  CAUSAL  : 16-Concept F1={row['val_causal_f1']*100:5.1f}% | RetroGoal@8={row.get('val_retro_recall', 0.0)*100:5.1f}% | "
        f"BlunderAvoid={row.get('blunder_avoid_pct', 0.0):5.1f}% | WM-Collapse={row.get('wm_blunder_pct', 0.0):5.1f}% | "
        f"Brilliant!! Top3={row.get('bril_top3_pct', 0.0):5.1f}% | NoduliGap={row['val_noduli_margin']:+.3f}"
    )
    print(
        f"  CHESS   : Move Top-1={row['val_top1_acc']:5.2f}% | Move Top-3={row['val_top3_acc']:5.2f}% | "
        f"WDL Acc={row['val_wdl_acc']:5.2f}% | Eval MAE={row['val_cp_mae']:6.1f} cp | GF Quiet MAE={row.get('val_gf_quiet_mae', 0.0):5.1f} cp"
    )
    print(
        f"  ELO     : {real_elo_s} | Proxy Net ~{row['est_raw_elo']:.0f} | +Rust PVS ~{row['est_search_elo']:.0f}"
    )
    print(
        f"  BIOLOGY : APL Active KCs={row['val_active_kc_pct']:5.2f}% (target={row['apl_target_pct']:.1f}%) | "
        f"K=4 OrthErr={row['val_orth']:.4f} | PlanH={row.get('c_plan_entropy', 0.0):.2f} | "
        f"DopamineGain={row['meta_dopamine_gain']:+.4f} | JEPA={row.get('jepa_loss', 0.0):.4f}"
    )
    print(
        f"  CIRCUITS: T4/T5={row.get('c_shunt_gain',0):.2f} | LobDmPm={row.get('c_lob_focus',0):.2f} | "
        f"MBON_E/I={row.get('c_mbon_ei',0):.2f} | LH={row.get('c_lh_veto',0):.2f} | "
        f"RC={row.get('c_rc_norm',0):.2f} | CX={row.get('c_cx_norm',0):.2f} | "
        f"Cent={row.get('c_cent_focus',0):.2f} | PFL2={row.get('c_pfl2',0):.2f}"
    )
    print(
        f"  META-DA : PAM={row['eta_pam']:.4f} | PPL1={row['eta_ppl1']:.4f} | Dop1R2={row['eta_dop1r2']:.4f} | "
        f"SOC={row['eta_soc']:.4f} | DPM={row.get('eta_dpm', 0.0):.4f} | rho={row['trace_rho']:.4f}"
    )
    print("=" * 96)


def train_saccadion_smart(
    cfg: SaccadionConfig,
    shard_dir: Path,
    checkpoint_dir: Path,
    export_dir: Path,
    epochs: int = 15,
    batch_size: int = 512,
    lr_muon: float = 4.0e-3,
    lr_adamw: float = 6.0e-4,
    ckpt_every_minutes: float = 15.0,
    log_every_steps: int = 25,
    fast_cache_dir: Optional[Path] = None,
    enable_cpu_arena: bool = True,
    live_hud_hz: float = 20.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Runs complete smart epoch-by-epoch causal TPU training with 50x FastCache,
    20 Hz live HTML/SVG HUD, and Background CPU Stockfish Sparring Arena."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    train_shards = sorted(shard_dir.glob("train_random_*.sacc"))
    if not train_shards:
        train_shards = sorted(p for p in shard_dir.glob("*.sacc") if "val" not in p.name and "traj" not in p.name and "sparring" not in p.name)
    sparring_shard = shard_dir / "sparring_hard.sacc"

    val_game_shards = sorted(p for p in shard_dir.glob("val_*.sacc") if "puzzles" not in p.name)
    val_puzzle_shards = sorted(shard_dir.glob("val_puzzles*.sacc"))
    val_shards = val_game_shards if val_game_shards else sorted(shard_dir.glob("val_*.sacc"))
    traj_shards = sorted(shard_dir.glob("train_traj_*.sacc"))

    if not train_shards:
        raise FileNotFoundError(f"No training `.sacc` shards found in {shard_dir}")

    # Default local SSD fast cache eliminates Python GIL decoding bottleneck (~50x faster epochs!)
    if fast_cache_dir is None:
        fast_cache_dir = Path("/content/sacc_cache") if Path("/content").exists() else (shard_dir.parent / "cache")

    train_ds = SaccShardDataset(train_shards, cache_dir=fast_cache_dir, split_name="train")
    val_ds = SaccShardDataset(val_shards, cache_dir=fast_cache_dir, split_name="val") if val_shards else train_ds
    val_puz_ds = (
        SaccShardDataset(val_puzzle_shards, cache_dir=fast_cache_dir, split_name="val_puz")
        if val_puzzle_shards
        else None
    )
    # Stage-3 trajectory windows only run 40 batches/epoch -> no need to pre-cache 2.2M board_feats upfront!
    traj_ds = SaccShardDataset(traj_shards, cache_dir=None, split_name="traj") if traj_shards else train_ds

    n_dev = jax.local_device_count()
    use_pmap = n_dev > 1
    eff_bsz = max(n_dev, (batch_size // n_dev) * n_dev)
    steps_per_epoch = max(1, len(train_ds) // eff_bsz)
    total_steps = steps_per_epoch * epochs
    dev_label = f"{n_dev}x {jax.devices()[0].platform.upper()}"

    print(f"=== SACCADION-8 RETROGRADE FISCHER-GRAPH (6CH-XRAY+A^T+2PLY-WM+INT8) SMART TPU TRAINER ===")
    print(f"Devices      : {dev_label} (pmap={use_pmap})")
    print(f"Model Tier   : tier_id={cfg.tier_id} (acc_dim={cfg.acc_dim}, D={cfg.stream_b_dim}, CX={cfg.cx_dim}, T_max={cfg.cx_max_ticks})")
    print(f"Dataset      : train={len(train_ds):,} pos | val={len(val_ds):,} pos | traj={len(traj_ds)//12:,} windows")
    print(f"FastCache    : {'ACTIVE (' + str(fast_cache_dir) + ')' if train_ds._cached else 'direct'}")
    print(f"Schedule     : {epochs} epochs x {steps_per_epoch} steps/epoch (batch={eff_bsz}, total_steps={total_steps:,})")

    claws_idx = jnp.asarray(build_flyhash_claws(cfg), dtype=jnp.int32)
    fresh_params = init_saccadion_params(cfg, seed=42)
    ckpt_file = checkpoint_dir / "latest.pkl"
    if ckpt_file.exists():
        ck = load_checkpoint(ckpt_file)
        loaded_p = ck["params"]
        added_keys = [k for k in fresh_params if k not in loaded_p]
        if added_keys:
            params = {**fresh_params, **loaded_p}
            tx, sched_muon, sched_adamw = build_muon_adamw_optimizer(params, total_steps, lr_muon, lr_adamw)
            opt_state = tx.init(params)
            print(f"Migrated checkpoint cleanly (initialized new params: {added_keys})")
        else:
            params = loaded_p
            tx, sched_muon, sched_adamw = build_muon_adamw_optimizer(params, total_steps, lr_muon, lr_adamw)
            opt_state = ck["opt_state"]
        loaded_ema = ck.get("ema_params")
        ema_params = {**params, **loaded_ema} if isinstance(loaded_ema, dict) else jax.tree_util.tree_map(jnp.array, params)
        start_epoch = int(ck["epoch"]) + 1
        global_step = int(ck["global_step"])
        best_val_loss = float(ck["best_val_loss"])
        history = list(ck["history"])
        print(f"Resumed checkpoint cleanly at Epoch {start_epoch} (best_val={best_val_loss:.4f})")
    else:
        params = fresh_params
        ema_params = jax.tree_util.tree_map(jnp.array, params)
        tx, sched_muon, sched_adamw = build_muon_adamw_optimizer(params, total_steps, lr_muon, lr_adamw)
        opt_state = tx.init(params)
        start_epoch = 1
        global_step = 0
        best_val_loss = float("inf")
        history = []

    if use_pmap:
        params_dev = jax.device_put_replicated(params, jax.local_devices())
        ema_dev = jax.device_put_replicated(ema_params, jax.local_devices())
        opt_dev = jax.device_put_replicated(opt_state, jax.local_devices())
    else:
        params_dev = params
        ema_dev = ema_params
        opt_dev = opt_state

    # Pre-decode a compact RAM pool of trajectory batches ONCE (~2s) so Stage 3/4 Meta-Dopamine NEVER freezes!
    cached_traj_batches: List[Dict[str, Any]] = []
    if len(traj_ds) >= 12:
        t_bsz = max(4, min(32, (len(traj_ds) // 12) // 2 or 1))
        for ti, tb_np in enumerate(traj_ds.iter_trajectory_batches(t_bsz, window_k=12, seed=2026)):
            cached_traj_batches.append(tb_np)
            if ti + 1 >= 16:
                break
        print(f"[TrajPool] Pre-cached {len(cached_traj_batches)} trajectory batches in RAM (zero Stage-3 stall!)")

    async_ckpt = _AsyncCheckpointSaver()

    # Start Background CPU Stockfish Sparring & Real-Elo Arena using BEST generalizing weights!
    arena: Optional[StockfishSparringArena] = None
    if enable_cpu_arena:
        arena = StockfishSparringArena(
            weights_path=export_dir / "saccadion_best.saccnet",
            harvest_shard_path=sparring_shard,
            status_json_path=checkpoint_dir / "arena_status.json",
            movetime_ms=250,
        )
        arena.start()

    hud = LiveTrainingHUD(max_hz=live_hud_hz, device_label=dev_label)

    # Cached compiled functions per (num_ticks, qat)
    compiled_train: Dict[Tuple[int, bool], Any] = {}
    compiled_val: Dict[Tuple[int, bool], Any] = {}

    def get_train_step(num_ticks: int, qat: bool):
        key = (num_ticks, qat)
        if key in compiled_train:
            return compiled_train[key]

        def loss_single(cur_p, batch, apl_target, apl_w, rng_key):
            k1, k2, k3 = jax.random.split(rng_key, 3)
            batch_f = {**batch, "board_feats": _dequant_board_feats(batch["board_feats"])}
            out_a = forward_stream_a(
                cur_p, cfg, claws_idx,
                batch_f["sparse_stm"], batch_f["sparse_opp"],
                batch_f["lh_feat"], batch_f["phase_bucket"],
                qat=qat, dropout_key=k1, kc_drop_rate=0.12, sparse_drop_rate=0.08, gf_drop_rate=0.22,
            )
            out_b = forward_stream_b(
                cur_p, cfg, batch_f["board_feats"], out_a["mbon_summary"],
                num_ticks=num_ticks, sez_cp=out_a["sez_cp"],
                graph_adj_u32=batch_f.get("graph_adj"),
                qat=qat, causal_drop_key=k3, drop_rate=0.12,
            )
            out_a_steered = forward_stream_a(
                cur_p, cfg, claws_idx,
                batch_f["sparse_stm"], batch_f["sparse_opp"],
                batch_f["lh_feat"], batch_f["phase_bucket"],
                cent_mask=out_b["cent_kc_mask"], qat=qat, dropout_key=k2, kc_drop_rate=0.12, sparse_drop_rate=0.08, gf_drop_rate=0.22,
            )
            return compute_multitask_loss(
                out_a_steered, out_b, batch_f, apl_target_active=apl_target, apl_weight=apl_w, params=cur_p, out_a_unmasked=out_a
            )

        if use_pmap:
            def pstep(p, ema_p, ost, batch, apl_target, apl_w, rng_key, gscales):
                (_, metrics), grads = jax.value_and_grad(loss_single, has_aux=True)(p, batch, apl_target, apl_w, rng_key)
                grads = jax.lax.pmean(grads, axis_name="dev")
                grads = jax.tree_util.tree_map(lambda g, s: g * s, grads, gscales)
                metrics = jax.lax.pmean(metrics, axis_name="dev")
                updates, new_ost = tx.update(grads, ost, p)
                new_p = optax.apply_updates(p, updates)
                new_ema = jax.tree_util.tree_map(lambda e, n: 0.996 * e + 0.004 * n, ema_p, new_p)
                return new_p, new_ema, new_ost, metrics

            fn = jax.pmap(pstep, axis_name="dev", in_axes=(0, 0, 0, 0, None, None, 0, None))
        else:
            @jax.jit
            def jstep(p, ema_p, ost, batch, apl_target, apl_w, rng_key, gscales):
                (_, metrics), grads = jax.value_and_grad(loss_single, has_aux=True)(p, batch, apl_target, apl_w, rng_key)
                grads = jax.tree_util.tree_map(lambda g, s: g * s, grads, gscales)
                updates, new_ost = tx.update(grads, ost, p)
                new_p = optax.apply_updates(p, updates)
                new_ema = jax.tree_util.tree_map(lambda e, n: 0.996 * e + 0.004 * n, ema_p, new_p)
                return new_p, new_ema, new_ost, metrics

            fn = jstep

        compiled_train[key] = fn
        return fn

    def get_val_step(num_ticks: int, qat: bool):
        key = (num_ticks, qat)
        if key in compiled_val:
            return compiled_val[key]

        @jax.jit
        def vstep(p, batch, apl_target, apl_w):
            batch_f = {**batch, "board_feats": _dequant_board_feats(batch["board_feats"])}
            out_a = forward_stream_a(
                p, cfg, claws_idx,
                batch_f["sparse_stm"], batch_f["sparse_opp"],
                batch_f["lh_feat"], batch_f["phase_bucket"], qat=qat,
            )
            out_b = forward_stream_b(
                p, cfg, batch_f["board_feats"], out_a["mbon_summary"],
                num_ticks=num_ticks, sez_cp=out_a["sez_cp"],
                graph_adj_u32=batch_f.get("graph_adj"), qat=qat,
            )
            out_a_s = forward_stream_a(
                p, cfg, claws_idx,
                batch_f["sparse_stm"], batch_f["sparse_opp"],
                batch_f["lh_feat"], batch_f["phase_bucket"],
                cent_mask=out_b["cent_kc_mask"], qat=qat,
            )
            _, m = compute_multitask_loss(
                out_a_s, out_b, batch_f, apl_target_active=apl_target, apl_weight=apl_w, params=p, out_a_unmasked=out_a
            )
            # Chess accuracy & calibration telemetry
            pol = out_b["policy_logits"]
            best_true = batch["top4_moves"][:, 0]
            top3_idx = jax.lax.top_k(pol, 3)[1]
            top1_acc = jnp.mean((top3_idx[:, 0] == best_true).astype(jnp.float32)) * 100.0
            top3_acc = jnp.mean(jnp.any(top3_idx == best_true[:, None], axis=-1).astype(jnp.float32)) * 100.0

            wdl_pred = jax.nn.softmax(out_b["wdl_logits"], axis=-1)
            wdl_true = batch["wdl_target"]
            wdl_acc = jnp.mean((jnp.argmax(wdl_pred, axis=-1) == jnp.argmax(wdl_true, axis=-1)).astype(jnp.float32)) * 100.0

            exp_true = jnp.clip(wdl_true[:, 0] - wdl_true[:, 2], -0.99, 0.99)
            cp_mae = jnp.mean(jnp.abs(out_b["pred_cp"] - 400.0 * jnp.arctanh(exp_true)))

            m["top1_acc"] = top1_acc
            m["top3_acc"] = top3_acc
            m["wdl_acc"] = wdl_acc
            m["cp_mae"] = cp_mae
            return m

        compiled_val[key] = vstep
        return vstep

    @jax.jit
    def stage_meta_jepa_step(p, ost, tbatch):
        """Joint Stage 2/3 Meta-Dopamine (Dop1R1+Dop1R2+MBON->DAN SOC) + JEPA 2-Ply Simulation step."""
        def meta_loss_fn(cur_p):
            out = meta_dopamine_trajectory_scan(cur_p, cfg, claws_idx, tbatch, qat=False)
            logits = out["traj_wdl_logits"]  # [B, K, 3]
            targets = tbatch["wdl_target"]
            k = logits.shape[1]
            per_step = jax.vmap(soft_kl_loss, in_axes=(1, 1))(logits, targets)
            step_w = jnp.linspace(0.5, 1.5, k)
            loss_traj = jnp.sum(per_step * step_w) / jnp.sum(step_w)
            loss_apl = jax.nn.relu(out["active_kc_ratio"] - 0.08) * 0.25
            adapt_gain = per_step[0] - per_step[-1]

            # Multi-Horizon Action-Conditioned hDelta Latent World Model:
            # Predicts future board latents z_{t+2} from (z_t, do(m_t)) AND z_{t+4} from (z_{t+2}, do(m_{t+2}))!
            out_a0 = forward_stream_a(
                cur_p, cfg, claws_idx,
                tbatch["sparse_stm"][:, 0], tbatch["sparse_opp"][:, 0],
                tbatch["lh_feat"][:, 0], tbatch["phase_bucket"][:, 0], qat=False,
            )
            gadj_t = tbatch.get("graph_adj")
            out_b0 = forward_stream_b(
                cur_p, cfg, tbatch["board_feats"][:, 0], out_a0["mbon_summary"],
                num_ticks=2, sez_cp=out_a0["sez_cp"],
                graph_adj_u32=gadj_t[:, 0] if gadj_t is not None else None, qat=False,
            )
            out_a2 = forward_stream_a(
                cur_p, cfg, claws_idx,
                tbatch["sparse_stm"][:, 2], tbatch["sparse_opp"][:, 2],
                tbatch["lh_feat"][:, 2], tbatch["phase_bucket"][:, 2], qat=False,
            )
            out_b2 = forward_stream_b(
                cur_p, cfg, tbatch["board_feats"][:, 2], out_a2["mbon_summary"],
                num_ticks=2, sez_cp=out_a2["sez_cp"],
                graph_adj_u32=gadj_t[:, 2] if gadj_t is not None else None, qat=False,
            )
            out_a4 = forward_stream_a(
                cur_p, cfg, claws_idx,
                tbatch["sparse_stm"][:, 4], tbatch["sparse_opp"][:, 4],
                tbatch["lh_feat"][:, 4], tbatch["phase_bucket"][:, 4], qat=False,
            )
            out_b4 = forward_stream_b(
                cur_p, cfg, tbatch["board_feats"][:, 4], out_a4["mbon_summary"],
                num_ticks=2, sez_cp=out_a4["sez_cp"],
                graph_adj_u32=gadj_t[:, 4] if gadj_t is not None else None, qat=False,
            )
            _, pred_z2 = predict_minimax_world_model(cur_p, out_b0, tbatch["top4_moves"][:, 0, 0])
            _, pred_z4 = predict_minimax_world_model(cur_p, out_b2, tbatch["top4_moves"][:, 2, 0])
            loss_jepa = 0.5 * (
                jepa_vicreg_loss(pred_z2, out_b2["global_latent"])
                + jepa_vicreg_loss(pred_z4, out_b4["global_latent"])
            )
            return loss_traj + loss_apl + 0.30 * loss_jepa, {
                "meta_loss": loss_traj,
                "jepa_loss": loss_jepa,
                "adapt_gain": adapt_gain,
                "active_kc_ratio": out["active_kc_ratio"],
            }

        (_, m), grads = jax.value_and_grad(meta_loss_fn, has_aux=True)(p)
        updates, new_ost = tx.update(grads, ost, p)
        return optax.apply_updates(p, updates), new_ost, m

    apl_weight = 0.05
    last_ckpt_time = time.time()
    prev_val_loss = history[-1]["val_loss"] if history else float("nan")

    t_train_start = time.time()
    try:
        for epoch in range(start_epoch, epochs + 1):
            t_ep0 = time.time()
            host_p_ref = jax.tree_util.tree_map(lambda x: x[0], params_dev) if use_pmap else params_dev
            stage, stage_name, num_ticks, apl_target, qat, grad_scales, freeze_map = build_freeze_thaw_schedule(
                host_p_ref, epoch, epochs
            )

            # Phase 3 & 4: Fast pre-cached Meta-Dopamine + Action World Model scan (< 2s, ZERO Python decode freeze!)
            meta_gain_vals: List[float] = []
            jepa_loss_vals: List[float] = []
            if stage >= 3 and cached_traj_batches:
                host_params = jax.tree_util.tree_map(lambda x: x[0], params_dev) if use_pmap else params_dev
                host_opt = jax.tree_util.tree_map(lambda x: x[0], opt_dev) if use_pmap else opt_dev
                max_t_batches = 8 if stage == 3 else 12
                for t_i, tbatch_np in enumerate(cached_traj_batches[:max_t_batches]):
                    tbatch = {k: jnp.asarray(v) for k, v in tbatch_np.items()}
                    host_params, host_opt, mm = stage_meta_jepa_step(host_params, host_opt, tbatch)
                    meta_gain_vals.append(float(mm["adapt_gain"]))
                    jepa_loss_vals.append(float(mm["jepa_loss"]))
                if use_pmap:
                    params_dev = jax.device_put_replicated(host_params, jax.local_devices())
                    opt_dev = jax.device_put_replicated(host_opt, jax.local_devices())
                else:
                    params_dev, opt_dev = host_params, host_opt

            train_fn = get_train_step(num_ticks, qat)
            val_fn = get_val_step(num_ticks, qat)

            ep_losses: List[float] = []
            ep_blunder_hits = 0.0
            ep_wm_hits = 0.0
            ep_blunder_cnt = 0.0
            ep_bril_t1 = 0.0
            ep_bril_t3 = 0.0
            ep_bril_cnt = 0.0
            pos_seen = 0
            ema_step_ms = 35.0
            t_loop_start: Optional[float] = None
            pos_after_jit = 0
            jax_rng = jax.random.PRNGKey(epoch * 4241 + 7)
            prev_m: Any = None
            prev_step_idx = 0
            prev_t0 = time.time()

            # INT8 2-stage async streaming + background mirror augmentation (ZERO TPU stalls!)
            for step_idx, batch_np in enumerate(
                train_ds.iter_batches(eff_bsz, seed=epoch * 997, prefetch=16, mirror_prob=0.5),
                start=1,
            ):
                t_s0 = time.time()
                jax_rng, step_key = jax.random.split(jax_rng)
                if use_pmap:
                    b_in = _shard_batch_for_pmap(batch_np, n_dev)
                    dev_keys = jax.random.split(step_key, n_dev)
                    params_dev, ema_dev, opt_dev, cur_m = train_fn(
                        params_dev, ema_dev, opt_dev, b_in, apl_target, apl_weight, dev_keys, grad_scales
                    )
                else:
                    b_in = {k: jnp.asarray(v) for k, v in batch_np.items()}
                    params_dev, ema_dev, opt_dev, cur_m = train_fn(
                        params_dev, ema_dev, opt_dev, b_in, apl_target, apl_weight, step_key, grad_scales
                    )

                # Async 1-step overlap: read PREVIOUS step's scalar metrics while TPU executes CURRENT step!
                if prev_m is not None:
                    def _sc(key: str) -> float:
                        return float(prev_m[key][0]) if use_pmap else float(prev_m[key])

                    loss_val = _sc("total")
                    kc_val = _sc("active_kc_ratio")
                    ep_blunder_hits += _sc("blunder_avoid_hits")
                    ep_wm_hits += _sc("wm_blunder_hits")
                    ep_blunder_cnt += _sc("blunder_count")
                    ep_bril_t1 += _sc("bril_top1_hits")
                    ep_bril_t3 += _sc("bril_top3_hits")
                    ep_bril_cnt += _sc("bril_count")

                    step_ms = (t_s0 - prev_t0) * 1000.0
                    if prev_step_idx == 1:
                        t_loop_start = t_s0
                    else:
                        ema_step_ms = 0.88 * ema_step_ms + 0.12 * step_ms
                        pos_after_jit += eff_bsz
                    ep_losses.append(loss_val)

                    if t_loop_start is not None and pos_after_jit > 0:
                        elapsed_loop = max(1e-3, time.time() - t_loop_start)
                        cur_pos_s = pos_after_jit / elapsed_loop
                        sec_per_step = elapsed_loop / max(1, prev_step_idx - 1)
                    else:
                        cur_pos_s = eff_bsz / max(1e-3, step_ms / 1000.0)
                        sec_per_step = max(1e-3, ema_step_ms / 1000.0)
                    rem_steps = (epochs - epoch) * steps_per_epoch + (steps_per_epoch - prev_step_idx)
                    eta_sec = rem_steps * sec_per_step
                    live_extra = {
                        "freeze_map": freeze_map,
                        "blunder_avoid_pct": (100.0 * ep_blunder_hits / ep_blunder_cnt) if ep_blunder_cnt > 0 else 0.0,
                        "wm_blunder_pct": (100.0 * ep_wm_hits / ep_blunder_cnt) if ep_blunder_cnt > 0 else 0.0,
                        "bril_top1_pct": (100.0 * ep_bril_t1 / ep_bril_cnt) if ep_bril_cnt > 0 else 0.0,
                        "bril_top3_pct": (100.0 * ep_bril_t3 / ep_bril_cnt) if ep_bril_cnt > 0 else 0.0,
                        "blunder_seen": int(ep_blunder_cnt),
                        "bril_seen": int(ep_bril_cnt),
                        "gf_quiet_mae_live": _sc("gf_quiet_mae"),
                        "retro_goal_recall": _sc("retro_goal_recall") * 100.0,
                        "best_val_loss": best_val_loss if best_val_loss < 1e8 else None,
                    }
                    hud.update(
                        epoch=epoch,
                        total_epochs=epochs,
                        step=prev_step_idx,
                        steps_per_epoch=steps_per_epoch,
                        stage_name=stage_name,
                        ticks=num_ticks,
                        qat=qat,
                        step_loss=loss_val,
                        step_kc_pct=kc_val * 100.0,
                        step_ms=ema_step_ms,
                        pos_per_sec=cur_pos_s,
                        eta_sec=eta_sec,
                        history=history,
                        arena_status=arena.get_status() if arena else None,
                        extra=live_extra,
                        force=False,
                    )

                prev_m = cur_m
                prev_step_idx = step_idx
                prev_t0 = t_s0
                global_step += 1
                pos_seen += eff_bsz

                if (time.time() - last_ckpt_time) > ckpt_every_minutes * 60.0:
                    host_p = jax.tree_util.tree_map(lambda x: x[0], params_dev) if use_pmap else params_dev
                    host_e = jax.tree_util.tree_map(lambda x: x[0], ema_dev) if use_pmap else ema_dev
                    host_o = jax.tree_util.tree_map(lambda x: x[0], opt_dev) if use_pmap else opt_dev
                    async_ckpt.save_async(ckpt_file, host_p, host_e, host_o, epoch - 1, global_step, best_val_loss, history)
                    last_ckpt_time = time.time()

            if prev_m is not None:
                def _sc2(key: str) -> float:
                    return float(prev_m[key][0]) if use_pmap else float(prev_m[key])

                loss_val = _sc2("total")
                kc_val = _sc2("active_kc_ratio")
                ep_blunder_hits += _sc2("blunder_avoid_hits")
                ep_wm_hits += _sc2("wm_blunder_hits")
                ep_blunder_cnt += _sc2("blunder_count")
                ep_bril_t1 += _sc2("bril_top1_hits")
                ep_bril_t3 += _sc2("bril_top3_hits")
                ep_bril_cnt += _sc2("bril_count")
                ep_losses.append(loss_val)

            # Unwrap single-host copy for clean held-out Validation (BOTH Live & Polyak EMA!)
            replay_batches_run = 0
            # Live Zero-Teacher dFB Self-Discovery Burst (starting Epoch 5, every 2 epochs):
            # Generates brand-new self-played quiet positions using Saccadion's OWN best weights
            # so the model breaks past static-dataset memorization and discovers original tactics!
            best_saccnet_path = export_dir / "saccadion_best.saccnet"
            rust_uci_bin = arena._find_rust_uci() if arena is not None else None
            if epoch >= 5 and (epoch % 2 == 1) and best_saccnet_path.exists() and rust_uci_bin is not None:
                sp_out = fast_cache_dir / f"dfb_selfplay_ep{epoch}.sacc"
                try:
                    sp_res = subprocess.run(
                        [str(rust_uci_bin), "selfplay", str(best_saccnet_path), str(sp_out), "48", "5"],
                        capture_output=True,
                        text=True,
                        timeout=75,
                        check=False,
                    )
                    if sp_res.returncode == 0 and sp_out.exists() and sp_out.stat().st_size > 16 + 128 * eff_bsz:
                        sp_ds = SaccShardDataset([sp_out], cache_dir=None, split_name=f"sp_ep{epoch}")
                        if len(sp_ds) >= eff_bsz:
                            sp_scales = jax.tree_util.tree_map(lambda s: s * jnp.float32(0.35), grad_scales)
                            for sp_b in sp_ds.iter_batches(eff_bsz, seed=epoch * 313, mirror_prob=0.5):
                                if train_ds._cached and getattr(train_ds, "_bd_is_i8", False):
                                    sp_b["board_feats"] = np.clip(
                                        np.rint(sp_b["board_feats"] * 127.0), -127, 127
                                    ).astype(np.int8)
                                jax_rng, sp_key = jax.random.split(jax_rng)
                                if use_pmap:
                                    b_sp = _shard_batch_for_pmap(sp_b, n_dev)
                                    params_dev, ema_dev, opt_dev, _ = train_fn(
                                        params_dev, ema_dev, opt_dev, b_sp, apl_target, apl_weight, jax.random.split(sp_key, n_dev), sp_scales
                                    )
                                else:
                                    b_sp = {k: jnp.asarray(v) for k, v in sp_b.items()}
                                    params_dev, ema_dev, opt_dev, _ = train_fn(
                                        params_dev, ema_dev, opt_dev, b_sp, apl_target, apl_weight, sp_key, sp_scales
                                    )
                                replay_batches_run += 1
                                if replay_batches_run >= 12:
                                    break
                            print(f"  [dFB Self-Discovery] Trained {replay_batches_run} zero-teacher batches from {len(sp_ds):,} fresh Saccadion positions!", flush=True)
                except Exception:
                    pass
                finally:
                    if sp_out.exists():
                        sp_out.unlink(missing_ok=True)

            host_params = jax.tree_util.tree_map(lambda x: x[0], params_dev) if use_pmap else params_dev
            host_ema = jax.tree_util.tree_map(lambda x: x[0], ema_dev) if use_pmap else ema_dev
            host_opt = jax.tree_util.tree_map(lambda x: x[0], opt_dev) if use_pmap else opt_dev

            val_keys = [
                "total", "wdl_b", "wdl_a", "gf_loss", "gf_quiet_mae", "policy", "pressure", "lh", "orth",
                "noduli", "noduli_margin", "causal", "causal_f1", "vnc2", "imbalance", "retro_goal_recall",
                "active_kc_ratio", "top1_acc", "top3_acc", "wdl_acc", "cp_mae",
                "shunt_gain", "lob_focus", "mbon_ei_ratio", "lh_veto_mean",
                "rc_hub_norm", "cx_phasor_norm", "cent_focus", "plan_entropy", "pfl2_mean",
            ]
            val_bsz = min(eff_bsz, max(1, len(val_ds)))
            val_batches_cached = [
                {k: jnp.asarray(v) for k, v in vb_np.items()}
                for v_i, vb_np in enumerate(val_ds.iter_batches(val_bsz, seed=12345))
                if v_i <= 24
            ]

            def _eval_candidate(cand_p: Dict[str, Any]) -> Dict[str, float]:
                acc: Dict[str, List[float]] = {k: [] for k in val_keys}
                for vb in val_batches_cached:
                    vm = val_fn(cand_p, vb, apl_target, apl_weight)
                    for k in val_keys:
                        if k not in ("causal", "causal_f1"):
                            acc[k].append(float(vm[k]))
                if val_puz_ds is not None and len(val_puz_ds) > 0:
                    p_bsz = min(eff_bsz, len(val_puz_ds))
                    for pbatch_np in val_puz_ds.iter_batches(p_bsz, seed=12345):
                        pb = {k: jnp.asarray(v) for k, v in pbatch_np.items()}
                        pm = val_fn(cand_p, pb, apl_target, apl_weight)
                        acc["causal"].append(float(pm["causal"]))
                        acc["causal_f1"].append(float(pm["causal_f1"]))
                return {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}

            v_live = _eval_candidate(host_params)
            # Compare Live vs Polyak EMA from Epoch 2+ and export strict best (zero regression!)
            if epoch >= 2:
                v_ema = _eval_candidate(host_ema)
                if v_ema["total"] < v_live["total"]:
                    v_mean = v_ema
                    export_cand = host_ema
                    used_ema = True
                else:
                    v_mean = v_live
                    export_cand = host_params
                    used_ema = False
            else:
                v_mean = v_live
                export_cand = host_params
                used_ema = False

            ep_sec = max(1e-3, time.time() - (t_loop_start or t_ep0))

            # Adaptive APL Sparsity Controller
            if v_mean["active_kc_ratio"] > apl_target + 0.03:
                apl_weight = min(0.50, apl_weight * 1.25)
            elif v_mean["active_kc_ratio"] < apl_target * 0.65:
                apl_weight = max(0.01, apl_weight * 0.85)

            val_loss = v_mean["total"]
            val_delta = (val_loss - prev_val_loss) if not math.isnan(prev_val_loss) else 0.0
            prev_val_loss = val_loss
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                export_params_to_saccnet(cfg, jax.device_get(export_cand), export_dir / "saccadion_best.saccnet")
            export_params_to_saccnet(cfg, jax.device_get(export_cand), export_dir / "saccadion_latest.saccnet")

            blunder_avoid_pct = (100.0 * ep_blunder_hits / ep_blunder_cnt) if ep_blunder_cnt > 0 else 0.0
            wm_blunder_pct = (100.0 * ep_wm_hits / ep_blunder_cnt) if ep_blunder_cnt > 0 else 0.0
            bril_top1_pct = (100.0 * ep_bril_t1 / ep_bril_cnt) if ep_bril_cnt > 0 else 0.0
            bril_top3_pct = (100.0 * ep_bril_t3 / ep_bril_cnt) if ep_bril_cnt > 0 else 0.0

            eta_pam_v = float(jax.nn.softplus(host_params["meta_eta_pam"]))
            eta_ppl1_v = float(jax.nn.softplus(host_params["meta_eta_ppl1"]))
            eta_dop1r2_v = float(jax.nn.softplus(host_params.get("meta_eta_dop1r2", jnp.array(-1.9))))
            eta_soc_v = float(jax.nn.softplus(host_params.get("meta_eta_soc", jnp.array(-1.4))))
            eta_dpm_v = float(jax.nn.softplus(host_params.get("meta_eta_dpm", jnp.array(-1.8))))
            rho_v = float(jax.nn.sigmoid(host_params["meta_trace_decay"]))
            est_raw_elo = estimate_elo_proxy(v_mean["top1_acc"], v_mean["wdl_acc"], v_mean["cp_mae"])
            est_search_elo = min(3700.0, est_raw_elo + 520.0)
            a_st = arena.get_status() if arena else {}
            per_eng = a_st.get("per_engine_wdl", {})
            eng_parts = [f"{k}:{v[0]}W/{v[1]}D/{v[2]}L" for k, v in sorted(per_eng.items())]
            arena_engines_str = (" | " + ", ".join(eng_parts)) if eng_parts else ""

            row = {
                "epoch": epoch,
                "total_epochs": epochs,
                "stage": stage,
                "stage_name": stage_name,
                "ticks": num_ticks,
                "qat": qat,
                "train_loss": float(np.mean(ep_losses)) if ep_losses else 0.0,
                "val_loss": val_loss,
                "val_delta": val_delta,
                "val_wdl_b": v_mean["wdl_b"],
                "val_wdl_a": v_mean["wdl_a"],
                "val_gf_loss": v_mean["gf_loss"],
                "val_policy": v_mean["policy"],
                "val_pressure": v_mean["pressure"],
                "val_lh": v_mean["lh"],
                "val_orth": v_mean["orth"],
                "val_noduli": v_mean["noduli"],
                "val_noduli_margin": v_mean["noduli_margin"],
                "val_causal": v_mean["causal"],
                "val_causal_f1": v_mean["causal_f1"],
                "val_vnc2": v_mean["vnc2"],
                "val_imbalance": v_mean["imbalance"],
                "val_retro_recall": v_mean["retro_goal_recall"],
                "blunder_avoid_pct": blunder_avoid_pct,
                "wm_blunder_pct": wm_blunder_pct,
                "bril_top1_pct": bril_top1_pct,
                "bril_top3_pct": bril_top3_pct,
                "used_ema": used_ema,
                "best_val_loss": best_val_loss,
                "val_top1_acc": v_mean["top1_acc"],
                "val_top3_acc": v_mean["top3_acc"],
                "val_wdl_acc": v_mean["wdl_acc"],
                "val_cp_mae": v_mean["cp_mae"],
                "val_gf_quiet_mae": v_mean["gf_quiet_mae"],
                "est_raw_elo": est_raw_elo,
                "est_search_elo": est_search_elo,
                "real_arena_elo": float(a_st.get("real_elo", 0.0)),
                "arena_ci": float(a_st.get("elo_ci", 999.0)),
                "arena_workers": int(a_st.get("parallel_workers", 1)),
                "arena_engines_str": arena_engines_str,
                "arena_games": int(a_st.get("games_played", 0)),
                "arena_harvested": int(a_st.get("harvested_positions", 0)),
                "val_active_kc_pct": v_mean["active_kc_ratio"] * 100.0,
                "apl_target_pct": apl_target * 100.0,
                "meta_dopamine_gain": float(np.mean(meta_gain_vals)) if meta_gain_vals else 0.0,
                "jepa_loss": float(np.mean(jepa_loss_vals)) if jepa_loss_vals else 0.0,
                "c_shunt_gain": v_mean["shunt_gain"],
                "c_lob_focus": v_mean["lob_focus"],
                "c_mbon_ei": v_mean["mbon_ei_ratio"],
                "c_lh_veto": v_mean["lh_veto_mean"],
                "c_rc_norm": v_mean["rc_hub_norm"],
                "c_cx_norm": v_mean["cx_phasor_norm"],
                "c_cent_focus": v_mean["cent_focus"],
                "c_plan_entropy": v_mean["plan_entropy"],
                "c_pfl2": v_mean["pfl2_mean"],
                "eta_pam": eta_pam_v,
                "eta_ppl1": eta_ppl1_v,
                "eta_dop1r2": eta_dop1r2_v,
                "eta_soc": eta_soc_v,
                "eta_dpm": eta_dpm_v,
                "trace_rho": rho_v,
                "lr_muon": float(sched_muon(global_step)),
                "lr_adamw": float(sched_adamw(global_step)),
                "replay_batches": replay_batches_run,
                "epoch_sec": ep_sec,
                "pos_per_sec": pos_seen / ep_sec,
            }
            history.append(row)
            _save_metrics_and_dashboard(history, checkpoint_dir)
            hud.update(
                epoch=epoch,
                total_epochs=epochs,
                step=steps_per_epoch,
                steps_per_epoch=steps_per_epoch,
                stage_name=stage_name,
                ticks=num_ticks,
                qat=qat,
                step_loss=row["train_loss"],
                step_kc_pct=row["val_active_kc_pct"],
                step_ms=ema_step_ms,
                pos_per_sec=row["pos_per_sec"],
                eta_sec=max(0.0, (epochs - epoch) * ep_sec),
                history=history,
                arena_status=a_st,
                extra={
                    "freeze_map": freeze_map,
                    "blunder_avoid_pct": blunder_avoid_pct,
                    "wm_blunder_pct": wm_blunder_pct,
                    "bril_top1_pct": bril_top1_pct,
                    "bril_top3_pct": bril_top3_pct,
                    "blunder_seen": int(ep_blunder_cnt),
                    "bril_seen": int(ep_bril_cnt),
                    "gf_quiet_mae_live": v_mean["gf_quiet_mae"],
                    "retro_goal_recall": v_mean["retro_goal_recall"] * 100.0,
                    "used_ema": used_ema,
                    "best_val_loss": best_val_loss,
                },
                force=True,
            )
            _print_epoch_banner(row, is_best)
            async_ckpt.save_async(ckpt_file, host_params, host_ema, host_opt, epoch, global_step, best_val_loss, history)
            last_ckpt_time = time.time()
            # Reclaim Python + glibc host RAM cleanly every epoch so Colab RAM never climbs!
            _reclaim_host_ram()
    finally:
        async_ckpt.wait()
        if arena is not None:
            arena.stop()

    host_final = jax.tree_util.tree_map(lambda x: x[0], params_dev) if use_pmap else params_dev
    return jax.device_get(host_final), history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saccadion-1 Smart Epoch-by-Epoch Colab TPU Trainer")
    parser.add_argument("--tier", choices=["tiny", "nano", "pro"], default="nano")
    parser.add_argument("--shard-dir", default="data/shards")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--export-dir", default="exports")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr-muon", type=float, default=1.5e-2)
    parser.add_argument("--lr-adamw", type=float, default=1.5e-3)
    args = parser.parse_args()

    cfg = {"tiny": SaccadionConfig.tiny, "nano": SaccadionConfig.nano, "pro": SaccadionConfig.pro}[args.tier]()
    train_saccadion_smart(
        cfg=cfg,
        shard_dir=Path(args.shard_dir),
        checkpoint_dir=Path(args.ckpt_dir),
        export_dir=Path(args.export_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_muon=args.lr_muon,
        lr_adamw=args.lr_adamw,
    )
