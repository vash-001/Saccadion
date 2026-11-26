"""8-Sensor Multi-Task Loss Function for Beyond-Stockfish Strategic & Creative Training.

Includes:
1. WDL Soft KL Divergence (Stream B Deep Supervision + Stream A + B->A Distillation)
2. Sparse Top-K PFL3 Policy Loss + 2-Ply VNC Synergy
3. 10-20 Ply Strategic Pressure Trajectory Huber Loss
4. 64-Square Optic ThreatMap Binary Cross-Entropy
5. PFL2 Tactical Volatility Regression
6. Lateral Horn 4-Channel Veto Loss (Blunder, Fortress, Mate, Passer)
7. JEPA 2-Ply Latent Mental Simulation + VICReg Anti-Collapse Regularizer
8. K=4 Creative Plan Orthogonality + APL 92% Exact-Sparsity Annealing
"""

from __future__ import annotations

from typing import Dict, Tuple
import jax
import jax.numpy as jnp


def soft_kl_loss(logits: jnp.ndarray, target_probs: jnp.ndarray) -> jnp.ndarray:
    log_p = jax.nn.log_softmax(logits, axis=-1)
    safe_t = jnp.clip(target_probs, 1e-6, 1.0)
    return jnp.mean(jnp.sum(safe_t * (jnp.log(safe_t) - log_p), axis=-1))


def huber_loss(pred: jnp.ndarray, target: jnp.ndarray, delta: float = 0.25) -> jnp.ndarray:
    err = jnp.abs(pred - target)
    return jnp.mean(jnp.where(err <= delta, 0.5 * err * err, delta * (err - 0.5 * delta)))


def sparse_topk_policy_loss(
    policy_logits: jnp.ndarray,  # [B, 4096]
    top4_moves: jnp.ndarray,     # [B, 4] int32
    top4_probs: jnp.ndarray,     # [B, 4] float32 (sums to 1.0)
    label_smoothing: float = 0.05,
) -> jnp.ndarray:
    log_p = jax.nn.log_softmax(policy_logits, axis=-1)
    gathered = jnp.take_along_axis(log_p, top4_moves, axis=-1)
    nll = -jnp.mean(jnp.sum(top4_probs * gathered, axis=-1))
    # Smooth entropy regularizer prevents overconfident logit memorization on 4096 classes
    smooth = -jnp.mean(log_p)
    return (1.0 - label_smoothing) * nll + label_smoothing * smooth


def creative_orthogonality_loss(plans: jnp.ndarray) -> jnp.ndarray:
    """Enforces K=4 Fan-Shaped Body strategic goal vectors to be mutually orthogonal."""
    # plans: [B, K, cx_d], unit-normalized
    gram = jnp.einsum("bkc,bjc->bkj", plans, plans)
    k = plans.shape[1]
    eye = jnp.eye(k, dtype=plans.dtype)[None, :, :]
    return jnp.mean((gram - eye) ** 2)


def jepa_vicreg_loss(pred_latent: jnp.ndarray, target_latent: jnp.ndarray) -> jnp.ndarray:
    """JEPA 2-ply latent prediction with Variance-Covariance anti-collapse term."""
    target_sg = jax.lax.stop_gradient(target_latent)
    inv_loss = jnp.mean((pred_latent - target_sg) ** 2)
    # Variance hinge preventing representation collapse
    std = jnp.sqrt(jnp.var(pred_latent, axis=0) + 1e-4)
    var_loss = jnp.mean(jax.nn.relu(0.5 - std))
    return inv_loss + 0.1 * var_loss


def counterfactual_noduli_loss(
    noduli_delta_map: jnp.ndarray,  # [B, 4096] in [-1, 1]
    policy_logits: jnp.ndarray,     # [B, 4096]
    best_move: jnp.ndarray,         # [B] int32
    wdl_target: jnp.ndarray,        # [B, 3]
    margin: float = 0.25,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Counterfactual Effectome Loss training Noduli `b_noduli_w` on `do(m*)` vs hard negatives.

    Forces the Noduli forward model to predict strictly higher causal advantage for the true
    winning move `do(m*)` than for the highest-scoring non-best counterfactual candidate moves.
    """
    bsz = noduli_delta_map.shape[0]
    row_idx = jnp.arange(bsz)
    pos_delta = noduli_delta_map[row_idx, best_move]  # [B]

    # Select top-3 hard-negative counterfactual moves (masking out true best_move)
    masked_pol = jax.lax.stop_gradient(policy_logits).at[row_idx, best_move].set(-1e9)
    neg_idx = jax.lax.top_k(masked_pol, 3)[1]  # [B, 3]
    neg_deltas = jnp.take_along_axis(noduli_delta_map, neg_idx, axis=-1)  # [B, 3]

    # Pairwise counterfactual margin loss: pos_delta >= neg_delta + margin
    margin_gap = pos_delta[:, None] - neg_deltas
    loss_margin = jnp.mean(jax.nn.relu(margin - margin_gap))
    # Calibrate positive move delta gently with true position expected value
    true_exp = wdl_target[:, 0] - wdl_target[:, 2]
    loss_cal = 0.25 * huber_loss(pos_delta, true_exp, delta=0.35)
    return loss_margin + loss_cal, jnp.mean(margin_gap)


def causal_concept_loss(
    causal_logits: jnp.ndarray,   # [B, 16]
    causal_targets: jnp.ndarray,  # [B, 16] in {0, 1}
    causal_mask: jnp.ndarray,     # [B] in {0, 1} (1 only on elite causal-tagged records)
    pos_weight: float = 4.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Positive-weighted masked multi-label BCE over the 16 Elite Causal Chess Concepts.

    Each puzzle activates ~2 of 16 concepts (12.5% positive rate); pos_weight=4.0 prevents
    trivial all-zero collapse and drives crisp recall/F1 across all 16 causal concepts!
    """
    log_sig = jax.nn.log_sigmoid(causal_logits)
    log_1msig = jax.nn.log_sigmoid(-causal_logits)
    bce = -(pos_weight * causal_targets * log_sig + (1.0 - causal_targets) * log_1msig)
    per_sample = jnp.mean(bce, axis=-1)  # [B]
    denom = jnp.maximum(jnp.sum(causal_mask), 1.0)
    loss = jnp.sum(per_sample * causal_mask) / denom

    # Soft F1 telemetry on tagged samples (threshold 0.45 calibrated for pos_weight=4)
    preds = (jax.nn.sigmoid(causal_logits) >= 0.45).astype(jnp.float32) * causal_mask[:, None]
    targs = causal_targets * causal_mask[:, None]
    tp = jnp.sum(preds * targs)
    fp = jnp.sum(preds * (1.0 - targs))
    fn = jnp.sum((1.0 - preds) * targs)
    f1 = (2.0 * tp) / jnp.maximum(2.0 * tp + fp + fn, 1e-5)
    return loss, f1


def compute_multitask_loss(
    out_a: Dict[str, jnp.ndarray],
    out_b: Dict[str, jnp.ndarray],
    batch: Dict[str, jnp.ndarray],
    apl_target_active: float = 0.08,
    apl_weight: float = 0.05,
    jepa_target_latent: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Computes full weighted 11-sensor causal multi-task loss and diagnostic scalar metrics."""
    wdl_t = batch["wdl_target"]

    # 1. Deep Supervision across all Central Complex deliberation ticks + Stream A + B->A Distill
    #    + Direct Centipawn Smooth-L1 Loss (non-saturating material & positional calibration up to +/-2500 cp!)
    ticks = out_b["per_tick_wdl"]
    n_t = len(ticks)
    w_sum = sum(range(1, n_t + 1))
    loss_wdl_b = sum(((i + 1) / w_sum) * soft_kl_loss(t_log, wdl_t) for i, t_log in enumerate(ticks))
    loss_wdl_a = soft_kl_loss(out_a["wdl_logits"], wdl_t)
    probs_b_sg = jax.lax.stop_gradient(jax.nn.softmax(out_b["wdl_logits"], axis=-1))
    loss_b_to_a = soft_kl_loss(out_a["wdl_logits"], probs_b_sg)

    target_exp = jnp.clip(wdl_t[:, 0] - wdl_t[:, 2], -0.995, 0.995)
    target_cp_scaled = jnp.arctanh(target_exp)  # target_cp / 400.0
    loss_cp_a = huber_loss(out_a["pred_cp"] / 400.0, target_cp_scaled, delta=0.5)
    loss_cp_b = huber_loss(out_b["pred_cp"] / 400.0, target_cp_scaled, delta=0.5)
    loss_wdl = loss_wdl_b + loss_wdl_a + 0.30 * loss_b_to_a + 0.45 * (loss_cp_a + loss_cp_b)

    # 2. Sparse Top-4 PFL3 Policy Loss
    loss_policy = sparse_topk_policy_loss(
        out_b["policy_logits"], batch["top4_moves"], batch["top4_probs"]
    )

    # 3. 10-20 Ply Strategic Pressure Trajectory Loss
    loss_pressure = huber_loss(out_b["strategic_pressure"], batch["pressure_targets"])

    # 4. 64-Square Optic ThreatMap Loss (on Visual Centrifugal board focus logits)
    threat_target = batch["threat_map_64"]  # [B, 64] in {0, 1}
    cent_p = jnp.clip(out_b["cent_board_mask"], 1e-5, 1.0 - 1e-5)
    loss_threat = -jnp.mean(
        threat_target * jnp.log(cent_p) + (1.0 - threat_target) * jnp.log(1.0 - cent_p)
    )

    # 5. PFL2 Tactical Volatility Loss
    loss_vol = huber_loss(out_b["pfl2_sharpness"], batch["volatility"])

    # 6. Lateral Horn 4-Channel Veto Loss
    loss_lh = huber_loss(out_a["lh_veto"], batch["lh_targets"])

    # 7. JEPA 2-Ply Mental Simulation Loss (if paired future latent provided)
    if jepa_target_latent is not None:
        loss_jepa = jepa_vicreg_loss(out_b["global_latent"], jepa_target_latent)
    else:
        loss_jepa = jnp.array(0.0, dtype=jnp.float32)

    # 8. K=4 Creative Plan Orthogonality + APL Exact-Sparsity Annealing
    loss_orth = creative_orthogonality_loss(out_b["creative_plans"])
    kc_density = 0.5 * (jnp.mean(out_a["kc_stm"]) + jnp.mean(out_a["kc_opp"]))
    loss_apl = jax.nn.relu(kc_density - apl_target_active)

    # 9. Counterfactual Noduli Efference-Copy Advantage Loss (trains b_noduli_w!)
    best_mv = batch["top4_moves"][:, 0]
    loss_noduli, noduli_margin = counterfactual_noduli_loss(
        out_b["noduli_delta_map"], out_b["policy_logits"], best_mv, wdl_t
    )

    # 10. AOTU -> LAL -> SMP 16-Elite-Causal-Concept Supervision (masked to tagged puzzles)
    if "causal_concepts" in batch and "causal_mask" in batch:
        loss_causal, causal_f1 = causal_concept_loss(
            out_b["causal_logits"], batch["causal_concepts"], batch["causal_mask"]
        )
    else:
        loss_causal = jnp.array(0.0, dtype=jnp.float32)
        causal_f1 = jnp.array(0.0, dtype=jnp.float32)

    # 11. 2-Ply VNC Causal Consequence Loss (predicts m_{t+2} enabled by m_t!)
    if "next_move" in batch:
        log_p2 = jax.nn.log_softmax(out_b["vnc2_logits"], axis=-1)
        next_mv = batch["next_move"]
        has_next = (next_mv != best_mv).astype(jnp.float32)
        nll2 = -log_p2[jnp.arange(log_p2.shape[0]), next_mv]
        loss_vnc2 = jnp.sum(nll2 * has_next) / jnp.maximum(jnp.sum(has_next), 1.0)
    else:
        loss_vnc2 = jnp.array(0.0, dtype=jnp.float32)

    total = (
        loss_wdl
        + 0.65 * loss_policy
        + 0.40 * loss_pressure
        + 0.30 * loss_threat
        + 0.20 * loss_vol
        + 0.15 * loss_lh
        + 0.25 * loss_jepa
        + 0.10 * loss_orth
        + apl_weight * loss_apl
        + 0.25 * loss_noduli
        + 0.65 * loss_causal
        + 0.12 * loss_vnc2
    )
    metrics = {
        "total": total,
        "wdl_b": loss_wdl_b,
        "wdl_a": loss_wdl_a,
        "policy": loss_policy,
        "pressure": loss_pressure,
        "lh": loss_lh,
        "orth": loss_orth,
        "noduli": loss_noduli,
        "noduli_margin": noduli_margin,
        "causal": loss_causal,
        "causal_f1": causal_f1,
        "vnc2": loss_vnc2,
        "active_kc_ratio": out_a["active_kc_ratio"],
        # All-12 Drosophila Brain Circuits live telemetry
        "shunt_gain": out_b["shunt_gain"],
        "lob_focus": out_b["lob_focus"],
        "mbon_ei_ratio": out_a["mbon_ei_ratio"],
        "lh_veto_mean": out_a["lh_veto_mean"],
        "rc_hub_norm": out_b["rc_hub_norm"],
        "cx_phasor_norm": out_b["cx_phasor_norm"],
        "cent_focus": out_b["cent_focus"],
        "plan_entropy": out_b["plan_entropy"],
        "pfl2_mean": out_b["pfl2_mean"],
    }
    return total, metrics
