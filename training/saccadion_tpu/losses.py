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
) -> jnp.ndarray:
    log_p = jax.nn.log_softmax(policy_logits, axis=-1)
    gathered = jnp.take_along_axis(log_p, top4_moves, axis=-1)
    return -jnp.mean(jnp.sum(top4_probs * gathered, axis=-1))


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


def compute_multitask_loss(
    out_a: Dict[str, jnp.ndarray],
    out_b: Dict[str, jnp.ndarray],
    batch: Dict[str, jnp.ndarray],
    apl_target_active: float = 0.08,
    apl_weight: float = 0.05,
    jepa_target_latent: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Computes full weighted 8-sensor loss and diagnostic scalar metrics."""
    wdl_t = batch["wdl_target"]

    # 1. Deep Supervision across all Central Complex deliberation ticks + Stream A + B->A Distill
    ticks = out_b["per_tick_wdl"]
    n_t = len(ticks)
    w_sum = sum(range(1, n_t + 1))
    loss_wdl_b = sum(((i + 1) / w_sum) * soft_kl_loss(t_log, wdl_t) for i, t_log in enumerate(ticks))
    loss_wdl_a = soft_kl_loss(out_a["wdl_logits"], wdl_t)
    probs_b_sg = jax.lax.stop_gradient(jax.nn.softmax(out_b["wdl_logits"], axis=-1))
    loss_b_to_a = soft_kl_loss(out_a["wdl_logits"], probs_b_sg)
    loss_wdl = loss_wdl_b + loss_wdl_a + 0.30 * loss_b_to_a

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
    # Differentiable L1 proxy pushing active KC fraction cleanly toward 8% (92% exact zeros!)
    kc_density = 0.5 * (jnp.mean(out_a["kc_stm"]) + jnp.mean(out_a["kc_opp"]))
    loss_apl = jax.nn.relu(kc_density - apl_target_active)

    total = (
        loss_wdl
        + 0.60 * loss_policy
        + 0.40 * loss_pressure
        + 0.30 * loss_threat
        + 0.20 * loss_vol
        + 0.15 * loss_lh
        + 0.25 * loss_jepa
        + 0.10 * loss_orth
        + apl_weight * loss_apl
    )
    metrics = {
        "total": total,
        "wdl_b": loss_wdl_b,
        "wdl_a": loss_wdl_a,
        "policy": loss_policy,
        "pressure": loss_pressure,
        "lh": loss_lh,
        "orth": loss_orth,
        "active_kc_ratio": out_a["active_kc_ratio"],
    }
    return total, metrics
