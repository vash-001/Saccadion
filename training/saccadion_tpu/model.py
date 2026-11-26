"""Complete Saccadion-1 Dual-Stream Neural Architecture in Pure JAX (TPU bfloat16/float32).

Implements ALL 9 empirically derived FlyWire v783 circuits identically to Rust `crates/saccadion-nn`:
- Stream A: Isolated Bilateral Feature Transformer + FlyHash 8-Claw Sparse Projection +
  APL Global GABA Inhibition (~92% zeros) + Push-Pull 2:1 MBONs + Lateral Horn 98.2%-I Veto +
  SMP/CRE/SIP Convergence Hub + Differentiable Eligibility-Trace PAM/12x-PPL1 Meta-Dopamine.
- Stream B: T4/T5 Shunting Inhibition ON-OFF Optic Encoder + Rich-Club Integrators/Broadcasters +
  Central Complex Phasor Engine (PFN + vDelta - hDelta 180-deg roll - Delta7 - Ring/Tangential) +
  Visual Centrifugal 71.8%-I Feedback + K=4 Orthogonal Creative Plan Bank +
  PFL3 99%-Contralateral Policy + PFL2 Sharpness + Noduli Efference + Pressure + Top-Down Seed.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .config import (
    AMACRINE_RANK,
    GRAPH_EDGE_TYPES,
    NUM_CAUSAL_CONCEPTS,
    OPTIC_LAYERS,
    PLASTIC_RANK,
    SaccadionConfig,
    build_flyhash_claws,
    init_sez_innate_weights,
)


def ste_quantize(w: jnp.ndarray, bits: int = 8, enabled: bool = False) -> jnp.ndarray:
    """Straight-Through Estimator (STE) symmetric fake-quantization for zero-loss Rust export."""
    if not enabled:
        return w
    qmax = float((1 << (bits - 1)) - 1)
    scale = jnp.maximum(jnp.max(jnp.abs(w)), 1e-6) / qmax
    w_q = jnp.round(jnp.clip(w / scale, -qmax, qmax)) * scale
    return w + jax.lax.stop_gradient(w_q - w)


def layer_norm(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + 1e-5) * gamma + beta


def init_saccadion_params(cfg: SaccadionConfig, seed: int = 42) -> Dict[str, jnp.ndarray]:
    """Initializes the complete Saccadion-1 parameter PyTree matching Rust SaccadionWeights layout."""
    key = jax.random.PRNGKey(seed)

    def u(shape: Tuple[int, ...], scale: float) -> jnp.ndarray:
        nonlocal key
        key, sub = jax.random.split(key)
        return jax.random.uniform(sub, shape, minval=-scale, maxval=scale, dtype=jnp.float32)

    _, gamma_dim, _ = cfg.lobe_dims_per_hemi()
    plastic_dim = cfg.plastic_dim_per_hemi()
    d = cfg.stream_b_dim
    cx = cfg.cx_dim
    mbon = cfg.mbon_dim
    stacks = cfg.num_phase_stacks
    smp_in = 2 * mbon + 4

    s_ft = float((1.0 / 32.0) ** 0.5 * 0.15)
    s_d = float((1.0 / d) ** 0.5 * 0.5)
    s_cx = float((1.0 / cx) ** 0.5 * 0.5)
    s_mbon = float((1.0 / (cfg.acc_dim * 0.1)) ** 0.5 * 0.25)

    return {
        # Stream A
        "a_ft_weight": u((cfg.sparse_input_dim, cfg.acc_dim), s_ft),
        "a_ft_bias": jnp.full((cfg.acc_dim,), 0.05, dtype=jnp.float32),
        "a_sez_weight": jnp.asarray(init_sez_innate_weights(cfg.sparse_input_dim, stacks), dtype=jnp.float32),
        "a_claw_weight": u((plastic_dim, cfg.flyhash_claws), 0.35),
        "a_apl_beta": jnp.array([0.85, 0.85], dtype=jnp.float32),
        "a_apl_thresh": jnp.full((cfg.acc_dim,), 0.18, dtype=jnp.float32),
        "a_mbon_ipsi": u((stacks, cfg.acc_dim, mbon), s_mbon),
        "a_mbon_contra": u((stacks, cfg.acc_dim, mbon), s_mbon),
        "a_mbon_bias": jnp.zeros((stacks, mbon), dtype=jnp.float32),
        "a_lh_w_inh": u((cfg.lh_hidden_dim, cfg.lh_input_dim), 0.15),
        "a_lh_b_inh": jnp.full((cfg.lh_hidden_dim,), 0.5, dtype=jnp.float32),
        "a_lh_w_out": u((4, cfg.lh_hidden_dim), 0.2),
        "a_lh_b_out": jnp.array([-1.2, -1.5, -1.5, -1.2], dtype=jnp.float32),
        "a_smp_w": u((stacks, 3, smp_in), float((1.0 / smp_in) ** 0.5 * 0.3)),
        "a_smp_b": jnp.zeros((stacks, 3), dtype=jnp.float32),
        # Stream B
        "b_shunt_exc": u((d, cfg.sq_feat_dim), s_d),
        "b_shunt_inh": u((d, cfg.sq_feat_dim), s_d),
        "b_shunt_leak": jnp.full((d,), 0.6, dtype=jnp.float32),
        "b_lc_w": u((d, cfg.sq_feat_dim), s_d),
        "b_optic_b": jnp.zeros((d,), dtype=jnp.float32),
        "b_hub_init": u((cfg.num_integrators + cfg.num_broadcasters, d), s_d),
        "b_int_q": u((d, d), s_d),
        "b_int_k": u((d, d), s_d),
        "b_int_v": u((d, d), s_d),
        "b_int_o": u((d, d), s_d),
        "b_rc_mix1": u((d, d), s_d),
        "b_rc_mix2": u((d, d), s_d),
        "b_rc_ln_g": jnp.ones((d,), dtype=jnp.float32),
        "b_rc_ln_b": jnp.zeros((d,), dtype=jnp.float32),
        "b_pfn_stm": u((cfg.cx_sectors, cx, d), s_d),
        "b_pfn_opp": u((cfg.cx_sectors, cx, d), s_d),
        "b_vdelta": u((cx, cx), s_cx),
        "b_hdelta": u((cx, cx), s_cx),
        "b_delta7": u((cx, cx), s_cx),
        "b_ring_inh": u((cx, d + mbon), s_d),
        "b_ring_b": jnp.zeros((cx,), dtype=jnp.float32),
        "b_cent_w": u((64, d), s_d),
        "b_cent_b": jnp.full((64,), 0.5, dtype=jnp.float32),
        "b_broad_to_sq": u((d, d), s_d),
        "b_sq_upd": u((d, d), s_d),
        "b_sq_ln_g": jnp.ones((d,), dtype=jnp.float32),
        "b_sq_ln_b": jnp.zeros((d,), dtype=jnp.float32),
        "b_plan_bank": u((cfg.num_creative_plans, cx, cx + d), s_d),
        "b_plan_b": jnp.zeros((cfg.num_creative_plans, cx), dtype=jnp.float32),
        "b_pfl3_q": u((cx, d), s_d),
        "b_pfl3_k": u((cx, d), s_d),
        "b_pfl3_opp": u((cx, d), s_d),
        "b_pfl2_w": u((d + cx,), s_d),
        "b_pfl2_b": jnp.array(0.0, dtype=jnp.float32),
        "b_wdl_w": u((3, d + cx), s_d),
        "b_wdl_b": jnp.zeros((3,), dtype=jnp.float32),
        "b_press_w": u((5, d + cx), s_d),
        "b_press_b": jnp.zeros((5,), dtype=jnp.float32),
        "b_noduli_w": u((cx,), s_cx),
        "b_noduli_b": jnp.array(0.0, dtype=jnp.float32),
        "b_seed_down": u((cfg.seed_rank, d + cx), s_d),
        "b_seed_to_gamma": u((gamma_dim, cfg.seed_rank), 0.1),
        "b_seed_to_cent": u((cfg.acc_dim, cfg.seed_rank), 0.1),
        # 3-Layer Untied FlyWire Optic Lobe Stack (Medulla -> Lobula -> Lobula Plate):
        # Each layer: Pre-LN -> Dm/Pm 4-Head 64x64 Attention -> Pre-LN -> Tm/TmY SwiGLU FFN!
        "b_opt_ln1_g": jnp.ones((OPTIC_LAYERS, d), dtype=jnp.float32),
        "b_opt_ln1_b": jnp.zeros((OPTIC_LAYERS, d), dtype=jnp.float32),
        "b_opt_q": u((OPTIC_LAYERS, d, d), s_d),
        "b_opt_k": u((OPTIC_LAYERS, d, d), s_d),
        "b_opt_v": u((OPTIC_LAYERS, d, d), s_d),
        "b_opt_o": u((OPTIC_LAYERS, d, d), s_d * 0.5),
        "b_opt_rel": jnp.zeros((OPTIC_LAYERS, 4, 64, 64), dtype=jnp.float32),
        "b_opt_ln2_g": jnp.ones((OPTIC_LAYERS, d), dtype=jnp.float32),
        "b_opt_ln2_b": jnp.zeros((OPTIC_LAYERS, d), dtype=jnp.float32),
        "b_opt_fg": u((OPTIC_LAYERS, 2 * d, d), s_d),
        "b_opt_fu": u((OPTIC_LAYERS, 2 * d, d), s_d),
        "b_opt_fd": u((OPTIC_LAYERS, d, 2 * d), s_d * 0.5),
        # Saccadion-5 Deep Fly-Brain Genius Circuits (exported to Rust SACCNET v5!)
        "a_plastic_to_mbon": u((mbon, PLASTIC_RANK), 0.15),
        "b_ama_u": u((OPTIC_LAYERS, 4 * AMACRINE_RANK, d), s_d * 0.5),
        "b_ama_v": u((OPTIC_LAYERS, 4 * AMACRINE_RANK, d), s_d * 0.5),
        "b_ama_g": u((OPTIC_LAYERS, 4 * AMACRINE_RANK, d), s_d * 0.5),
        "b_cx_ln_g": jnp.ones((cx,), dtype=jnp.float32),
        "b_cx_ln_b": jnp.zeros((cx,), dtype=jnp.float32),
        "b_fc2_conj": u((cx, 2 * cx), s_cx * 0.5),
        # Saccadion-6 FlyWire Typed Directed Chess-Graph GNN + Live hDelta World Model (exported to v6!)
        "b_gnn_ln_g": jnp.ones((d,), dtype=jnp.float32),
        "b_gnn_ln_b": jnp.zeros((d,), dtype=jnp.float32),
        "b_gnn_w": u((GRAPH_EDGE_TYPES, d, d), s_d * 0.5),
        "b_wm_w1": u((d + cx, d + 2 * cx), s_d),
        "b_wm_b1": jnp.zeros((d + cx,), dtype=jnp.float32),
        "b_wm_w2": u((d + cx, d + cx), s_d * 0.5),
        # Training-only AOTU -> LAL -> SMP 16-Causal-Concept Effectome Head (zero Rust latency!)
        "aux_causal_w": u((NUM_CAUSAL_CONCEPTS, d + cx), s_d),
        "aux_causal_b": jnp.full((NUM_CAUSAL_CONCEPTS,), -0.5, dtype=jnp.float32),
        # Training-only 2-Ply VNC Causal Consequence Head (predicts m_{t+2} enabled by m_t)
        "aux_vnc2_q": u((cx, d), s_d),
        "aux_vnc2_k": u((cx, d), s_d),
        # Learnable meta-plasticity scalars (Dop1R1 PAM/PPL1, trace rho, Dop1R2, SOC, DPM consolidation)
        "meta_eta_pam": jnp.array(-2.484, dtype=jnp.float32),
        "meta_eta_ppl1": jnp.array(-1.130, dtype=jnp.float32),
        "meta_trace_decay": jnp.array(0.619, dtype=jnp.float32),
        "meta_eta_dop1r2": jnp.array(-1.900, dtype=jnp.float32),
        "meta_eta_soc": jnp.array(-1.400, dtype=jnp.float32),
        "meta_eta_dpm": jnp.array(-1.800, dtype=jnp.float32),
    }


def forward_stream_a(
    params: Dict[str, jnp.ndarray],
    cfg: SaccadionConfig,
    claws_idx: jnp.ndarray,
    sparse_stm: jnp.ndarray,   # [B, max_active] int32 (-1 padded)
    sparse_opp: jnp.ndarray,   # [B, max_active] int32 (-1 padded)
    lh_feat: jnp.ndarray,      # [B, 32] float32
    phase_bucket: jnp.ndarray, # [B] int32 in 0..7
    cent_mask: jnp.ndarray | None = None,  # Optional [B, acc_dim] top-down inhibitory mask
    dw_gamma_stm: jnp.ndarray | None = None,  # Optional [B, gamma_dim, PLASTIC_RANK] fast matrix
    dw_gamma_opp: jnp.ndarray | None = None,
    dw_apbp_stm: jnp.ndarray | None = None,   # Optional [B, apbp_dim, PLASTIC_RANK] DPM matrix
    dw_apbp_opp: jnp.ndarray | None = None,
    octopamine: float = 1.0,
    qat: bool = False,
    dropout_key: Any | None = None,
    kc_drop_rate: float = 0.0,
) -> Dict[str, jnp.ndarray]:
    """Batched Stream A forward pass on TPU matching `crates/saccadion-nn/src/stream_a.rs`."""
    ab_dim, gamma_dim, _ = cfg.lobe_dims_per_hemi()
    half_mbon = cfg.mbon_dim // 2

    # 1. Isolated Bilateral Feature Accumulation + SEZ Innate Survival Valence (padded -1 masked out)
    ft_w = ste_quantize(params["a_ft_weight"], bits=16, enabled=qat)
    ft_pad = jnp.concatenate([ft_w, jnp.zeros((1, cfg.acc_dim), dtype=ft_w.dtype)], axis=0)
    sez_w = params["a_sez_weight"]
    sez_pad = jnp.concatenate([sez_w, jnp.zeros((1, cfg.num_phase_stacks), dtype=sez_w.dtype)], axis=0)
    pad_idx = cfg.sparse_input_dim
    idx_s = jnp.where(sparse_stm >= 0, sparse_stm, pad_idx)
    idx_o = jnp.where(sparse_opp >= 0, sparse_opp, pad_idx)
    acc_stm = jnp.sum(ft_pad[idx_s], axis=1) + params["a_ft_bias"]
    acc_opp = jnp.sum(ft_pad[idx_o], axis=1) + params["a_ft_bias"]

    bsz = sparse_stm.shape[0]
    row_arange = jnp.arange(bsz)
    sez_stm = jnp.sum(sez_pad[idx_s], axis=1)[row_arange, phase_bucket]
    sez_opp = jnp.sum(sez_pad[idx_o], axis=1)[row_arange, phase_bucket]
    sez_cp = (sez_stm - sez_opp) * 50.0

    # 2. FlyHash Sparse Claw Projection onto gamma + alpha'/beta' lobes
    claw_w = ste_quantize(params["a_claw_weight"], bits=8, enabled=qat)  # [plastic, claws]

    def apply_claws(acc: jnp.ndarray) -> jnp.ndarray:
        ab_part = acc[:, :ab_dim]
        gathered = acc[:, claws_idx]  # [B, plastic, claws]
        plastic_part = jnp.sum(gathered * claw_w[None, :, :], axis=-1)
        return jnp.concatenate([ab_part, plastic_part], axis=-1)

    pre_stm = apply_claws(acc_stm)
    pre_opp = apply_claws(acc_opp)

    # 3. APL Global GABA Inhibition Gate (per hemisphere, 99.9% ipsilateral)
    cent_inh = 0.25 * jnp.maximum(cent_mask, 0.0) if cent_mask is not None else 0.0
    mean_stm = jnp.mean(jnp.maximum(pre_stm, 0.0), axis=-1, keepdims=True)
    mean_opp = jnp.mean(jnp.maximum(pre_opp, 0.0), axis=-1, keepdims=True)
    kc_stm = jnp.clip(
        pre_stm - params["a_apl_beta"][0] * mean_stm - params["a_apl_thresh"] - cent_inh, 0.0, 1.0
    ) ** 2
    kc_opp = jnp.clip(
        pre_opp - params["a_apl_beta"][1] * mean_opp - params["a_apl_thresh"] - cent_inh, 0.0, 1.0
    ) ** 2

    active_kc_ratio = 0.5 * (jnp.mean((kc_stm > 0.0).astype(jnp.float32)) + jnp.mean((kc_opp > 0.0).astype(jnp.float32)))

    # Optional structured KC dropout during training only (prevents table memorization!)
    if dropout_key is not None and kc_drop_rate > 0.0:
        k1, k2 = jax.random.split(dropout_key)
        keep = 1.0 - kc_drop_rate
        m1 = (jax.random.uniform(k1, kc_stm.shape) < keep).astype(jnp.float32) / keep
        m2 = (jax.random.uniform(k2, kc_opp.shape) < keep).astype(jnp.float32) / keep
        kc_stm_eff = kc_stm * m1
        kc_opp_eff = kc_opp * m2
    else:
        kc_stm_eff = kc_stm
        kc_opp_eff = kc_opp

    # 4. Push-Pull MBONs (2:1 Ipsi / Contra mixing selected per game phase bucket)
    w_ipsi = ste_quantize(params["a_mbon_ipsi"], bits=8, enabled=qat)[phase_bucket]     # [B, acc, 32]
    w_contra = ste_quantize(params["a_mbon_contra"], bits=8, enabled=qat)[phase_bucket] # [B, acc, 32]
    b_mbon = params["a_mbon_bias"][phase_bucket]                                        # [B, 32]

    raw_mbon_stm = jnp.einsum("bi,bim->bm", kc_stm_eff, w_ipsi) + 0.5 * jnp.einsum("bi,bim->bm", kc_opp_eff, w_contra) + b_mbon
    raw_mbon_opp = jnp.einsum("bi,bim->bm", kc_opp_eff, w_ipsi) + 0.5 * jnp.einsum("bi,bim->bm", kc_stm_eff, w_contra) + b_mbon

    # True 3-Lobe Closed Cognitive Loop: inject active gamma + DPM alpha'/beta' rank-8 synapses
    # directly into the 32-dim MBON vector BEFORE Push-Pull SiLU and BEFORE Stream B CX!
    kc_gamma_stm = kc_stm[:, ab_dim : ab_dim + gamma_dim]
    kc_gamma_opp = kc_opp[:, ab_dim : ab_dim + gamma_dim]
    kc_apbp_stm = kc_stm[:, ab_dim + gamma_dim :]
    kc_apbp_opp = kc_opp[:, ab_dim + gamma_dim :]
    if dw_gamma_stm is not None and dw_gamma_opp is not None:
        lat_s = jnp.einsum("bg,bgr->br", kc_gamma_stm, dw_gamma_stm)
        lat_o = jnp.einsum("bg,bgr->br", kc_gamma_opp, dw_gamma_opp)
        if dw_apbp_stm is not None and dw_apbp_opp is not None:
            lat_s = lat_s + jnp.einsum("ba,bar->br", kc_apbp_stm, dw_apbp_stm)
            lat_o = lat_o + jnp.einsum("ba,bar->br", kc_apbp_opp, dw_apbp_opp)
        lat_s = jnp.tanh(lat_s) * 0.65
        lat_o = jnp.tanh(lat_o) * 0.65
        ds = jnp.einsum("br,mr->bm", lat_s, params["a_plastic_to_mbon"])
        do = jnp.einsum("br,mr->bm", lat_o, params["a_plastic_to_mbon"])
        raw_mbon_stm = raw_mbon_stm + ds - 0.5 * do
        raw_mbon_opp = raw_mbon_opp + do - 0.5 * ds

    signs = jnp.concatenate([jnp.ones((half_mbon,)), -jnp.ones((cfg.mbon_dim - half_mbon,))])
    mbon_stm = jax.nn.silu(raw_mbon_stm) * signs * octopamine
    mbon_opp = jax.nn.silu(raw_mbon_opp) * signs * octopamine

    # 5. Lateral Horn 98.2%-Inhibitory Innate Veto Circuit
    lh_w_inh = jnp.abs(ste_quantize(params["a_lh_w_inh"], bits=8, enabled=qat))
    lh_h = jax.nn.relu(params["a_lh_b_inh"] - jnp.einsum("bi,hi->bh", lh_feat, lh_w_inh))
    lh_raw = jax.nn.sigmoid(jnp.einsum("bh,oh->bo", lh_h, ste_quantize(params["a_lh_w_out"], 8, qat)) + params["a_lh_b_out"])
    lh_veto = jnp.stack(
        [
            jnp.clip(0.7 * lh_raw[:, 0] + 0.3 * lh_feat[:, 8], 0.0, 1.0),
            jnp.clip(0.65 * lh_raw[:, 1] + 0.35 * jnp.maximum(lh_feat[:, 17], lh_feat[:, 20]), 0.0, 1.0),
            jnp.clip(0.7 * lh_raw[:, 2] + 0.3 * jnp.minimum(lh_feat[:, 0] * lh_feat[:, 4], 1.0), 0.0, 1.0),
            jnp.clip(0.7 * lh_raw[:, 3] + 0.3 * jnp.maximum(lh_feat[:, 14], lh_feat[:, 15]), 0.0, 1.0),
        ],
        axis=-1,
    )

    # 6. SMP/CRE/SIP Convergence Hub -> Positional MBON residual + SEZ Innate Survival Fusion
    smp_in = jnp.concatenate([mbon_stm, mbon_opp, lh_veto], axis=-1)  # [B, 68]
    smp_w = ste_quantize(params["a_smp_w"], bits=8, enabled=qat)[phase_bucket]  # [B, 3, 68]
    smp_b = params["a_smp_b"][phase_bucket]                                     # [B, 3]
    wdl_logits_raw = jnp.einsum("bi,bki->bk", smp_in, smp_w) + smp_b

    mbon_probs = jax.nn.softmax(wdl_logits_raw, axis=-1)
    mbon_cp = 400.0 * jnp.arctanh(jnp.clip(mbon_probs[:, 0] - mbon_probs[:, 2], -0.999, 0.999))
    fortress_scale = 1.0 - 0.55 * lh_veto[:, 1]
    pred_cp = jnp.clip((sez_cp + mbon_cp) * fortress_scale, -2500.0, 2500.0)

    sez_l = jnp.clip(sez_cp / 280.0, -5.0, 5.0)
    wdl_logits = wdl_logits_raw + jnp.stack([sez_l, jnp.zeros_like(sez_l), -sez_l], axis=-1)

    mbon_summary = mbon_stm - 0.5 * mbon_opp
    mbon_e = jnp.mean(jnp.abs(mbon_stm[:, :half_mbon]))
    mbon_i = jnp.mean(jnp.abs(mbon_stm[:, half_mbon:]))
    return {
        "sez_cp": sez_cp,
        "pred_cp": pred_cp,
        "wdl_logits": wdl_logits,
        "lh_veto": lh_veto,
        "mbon_summary": mbon_summary,
        "kc_stm": kc_stm,
        "kc_opp": kc_opp,
        "kc_gamma_stm": kc_gamma_stm,
        "kc_gamma_opp": kc_gamma_opp,
        "kc_apbp_stm": kc_apbp_stm,
        "kc_apbp_opp": kc_apbp_opp,
        "active_kc_ratio": active_kc_ratio,
        "mbon_ei_ratio": mbon_e / (mbon_i + 1e-5),
        "lh_veto_mean": jnp.mean(lh_veto),
    }


def forward_stream_b(
    params: Dict[str, jnp.ndarray],
    cfg: SaccadionConfig,
    board_feats: jnp.ndarray,  # [B, 64, 48] float32
    mbon_summary: jnp.ndarray, # [B, 32] float32
    num_ticks: int,
    sez_cp: jnp.ndarray | None = None,
    graph_adj_u32: jnp.ndarray | None = None,  # [B, 4, 64, 2] uint32 FlyWire Chess-Graph adjacency
    serotonin: float = 1.0,
    qat: bool = False,
    causal_drop_key: Any | None = None,
) -> Dict[str, Any]:
    """Batched Stream B Deep Strategic Brain matching `crates/saccadion-nn/src/stream_b.rs`."""
    bsz = board_feats.shape[0]
    d = cfg.stream_b_dim
    n_int = cfg.num_integrators
    sectors = cfg.cx_sectors
    cx_d = cfg.cx_dim
    inv_sqrt_d = float(1.0 / (d ** 0.5))

    # B1. T4/T5 Shunting Inhibition Optic Encoder (computed safely in float32)
    w_exc = ste_quantize(params["b_shunt_exc"], 8, qat)
    w_inh = ste_quantize(params["b_shunt_inh"], 8, qat)
    w_lc = ste_quantize(params["b_lc_w"], 8, qat)
    exc = jax.nn.relu(jnp.einsum("bsf,df->bsd", board_feats, w_exc))
    inh = jax.nn.relu(jnp.einsum("bsf,df->bsd", board_feats, w_inh))
    leak = jax.nn.softplus(params["b_shunt_leak"]) + 0.5
    lc = jnp.einsum("bsf,df->bsd", board_feats, w_lc)
    sq_state = (exc.astype(jnp.float32) / (leak + inh.astype(jnp.float32))) + lc + params["b_optic_b"]

    # B1a. FlyWire Typed Directed Chess-Graph GNN (Dale's Law +ACh STM vs -GABA Opp)
    if graph_adj_u32 is not None:
        bit_shifts = jnp.arange(32, dtype=jnp.uint32)
        lo_bits = ((graph_adj_u32[..., 0:1] >> bit_shifts) & jnp.uint32(1)).astype(jnp.float32)
        hi_bits = ((graph_adj_u32[..., 1:2] >> bit_shifts) & jnp.uint32(1)).astype(jnp.float32)
        adj_f = jnp.concatenate([lo_bits, hi_bits], axis=-1)  # [B, 4, 64, 64]
        in_deg = jnp.sum(adj_f, axis=2, keepdims=True)        # [B, 4, 1, 64]
        adj_norm = adj_f / (in_deg + 1.0)
    else:
        adj_norm = None

    w_gnn = ste_quantize(params["b_gnn_w"], 8, qat)

    def apply_flywire_gnn_hop(cur_sq: jnp.ndarray) -> jnp.ndarray:
        if adj_norm is None:
            return cur_sq
        x_g = layer_norm(cur_sq, params["b_gnn_ln_g"], params["b_gnn_ln_b"])
        agg = jnp.einsum("bkuv,bud->bkvd", adj_norm, x_g)
        proj = jnp.einsum("bkvd,ked->bkve", agg, w_gnn)
        signed = proj[:, 0] + proj[:, 1] - proj[:, 2] - proj[:, 3]
        return cur_sq + jax.nn.silu(signed)

    # Hop 1: immediate 1-hop attack/defense synaptic propagation right after T4/T5 Shunting
    sq_state = apply_flywire_gnn_hop(sq_state)

    hubs = jnp.broadcast_to(params["b_hub_init"][None, :, :], (bsz, n_int + cfg.num_broadcasters, d))
    cx_phasor = jnp.zeros((bsz, sectors, cx_d), dtype=jnp.float32)

    w_q = ste_quantize(params["b_int_q"], 8, qat)
    w_k = ste_quantize(params["b_int_k"], 8, qat)
    w_v = ste_quantize(params["b_int_v"], 8, qat)
    w_o = ste_quantize(params["b_int_o"], 8, qat)
    w_rc1 = ste_quantize(params["b_rc_mix1"], 8, qat)
    w_rc2 = ste_quantize(params["b_rc_mix2"], 8, qat)
    w_pfn_s = ste_quantize(params["b_pfn_stm"], 8, qat)
    w_pfn_o = ste_quantize(params["b_pfn_opp"], 8, qat)
    w_vd = ste_quantize(params["b_vdelta"], 8, qat)
    w_hd = ste_quantize(params["b_hdelta"], 8, qat)
    w_d7 = ste_quantize(params["b_delta7"], 8, qat)
    w_ring = ste_quantize(params["b_ring_inh"], 8, qat)
    w_cent = ste_quantize(params["b_cent_w"], 8, qat)
    w_b2s = ste_quantize(params["b_broad_to_sq"], 8, qat)
    w_squpd = ste_quantize(params["b_sq_upd"], 8, qat)

    n_lob_h = 4
    lob_hd = d // n_lob_h
    inv_sqrt_hd = float(1.0 / (lob_hd ** 0.5))
    shunt_gain = jnp.mean(exc.astype(jnp.float32) / (leak + inh.astype(jnp.float32)))

    # B1b. 3-Layer Untied FlyWire Optic Lobe Stack (Medulla -> Lobula -> Lobula Plate):
    # Pre-LN -> Amacrine Dynamic Attention (Dm/Pm) -> Pre-LN -> Tm/TmY SwiGLU FFN
    inv_sqrt_ama = float(1.0 / (AMACRINE_RANK ** 0.5))
    lob_focus = jnp.array(0.0, dtype=jnp.float32)
    for l_idx in range(OPTIC_LAYERS):
        x1 = layer_norm(sq_state, params["b_opt_ln1_g"][l_idx], params["b_opt_ln1_b"][l_idx])
        wq_l = ste_quantize(params["b_opt_q"][l_idx], 8, qat)
        wk_l = ste_quantize(params["b_opt_k"][l_idx], 8, qat)
        wv_l = ste_quantize(params["b_opt_v"][l_idx], 8, qat)
        wo_l = ste_quantize(params["b_opt_o"][l_idx], 8, qat)
        rel_l = params["b_opt_rel"][l_idx]

        # Amacrine (Dm/Pm) Dynamic Position-Conditioned Relational Attention Gate
        x1_mean = jnp.mean(x1, axis=1)  # [B, D]
        ama_g = jax.nn.sigmoid(jnp.einsum("bd,rd->br", x1_mean, params["b_ama_g"][l_idx])).reshape(
            (bsz, n_lob_h, AMACRINE_RANK)
        )
        ama_u = jnp.tanh(jnp.einsum("bsd,rd->bsr", x1, params["b_ama_u"][l_idx])).reshape(
            (bsz, 64, n_lob_h, AMACRINE_RANK)
        )
        ama_v = jnp.tanh(jnp.einsum("bsd,rd->bsr", x1, params["b_ama_v"][l_idx])).reshape(
            (bsz, 64, n_lob_h, AMACRINE_RANK)
        )
        ama_bias = jnp.einsum("buhr,bhr,bvhr->bhuv", ama_u, ama_g, ama_v) * inv_sqrt_ama

        lq = jnp.einsum("bsd,ed->bse", x1, wq_l).reshape((bsz, 64, n_lob_h, lob_hd))
        lk = jnp.einsum("bsd,ed->bse", x1, wk_l).reshape((bsz, 64, n_lob_h, lob_hd))
        lv = jnp.einsum("bsd,ed->bse", x1, wv_l).reshape((bsz, 64, n_lob_h, lob_hd))
        l_scores = jnp.einsum("buhd,bvhd->bhuv", lq, lk) * inv_sqrt_hd + rel_l[None, :, :, :] + ama_bias
        l_attn = jax.nn.softmax(l_scores, axis=-1)
        lob_focus = jnp.mean(jnp.max(l_attn, axis=-1))
        l_ctx = jnp.einsum("bhuv,bvhd->buhd", l_attn, lv).reshape((bsz, 64, d))
        sq_state = sq_state + jnp.einsum("bse,de->bsd", l_ctx, wo_l)

        x2 = layer_norm(sq_state, params["b_opt_ln2_g"][l_idx], params["b_opt_ln2_b"][l_idx])
        wfg_l = ste_quantize(params["b_opt_fg"][l_idx], 8, qat)
        wfu_l = ste_quantize(params["b_opt_fu"][l_idx], 8, qat)
        wfd_l = ste_quantize(params["b_opt_fd"][l_idx], 8, qat)
        fg = jax.nn.silu(jnp.einsum("bsd,fd->bsf", x2, wfg_l))
        fu = jnp.einsum("bsd,fd->bsf", x2, wfu_l)
        sq_state = sq_state + jnp.einsum("bsf,df->bsd", fg * fu, wfd_l)
        # Hop 2 after Layer 0: 2-hop transitive battery/pin/overload propagation across chess graph!
        if l_idx == 0:
            sq_state = apply_flywire_gnn_hop(sq_state)

    # Weight-tied Central Complex + Rich-Club deliberation loop
    sez_vec = sez_cp if sez_cp is not None else jnp.zeros((bsz,), dtype=jnp.float32)
    sez_l = jnp.clip(sez_vec / 280.0, -5.0, 5.0)
    sez_shift = jnp.stack([sez_l, jnp.zeros_like(sez_l), -sez_l], axis=-1)
    per_tick_wdl = []
    last_wdl_raw = None
    for _ in range(num_ticks):
        # Step A: Integrators Cross-Attention from 64 squares
        int_h = hubs[:, :n_int, :]
        q = jnp.einsum("bhd,ed->bhe", int_h, w_q)
        k = jnp.einsum("bsd,ed->bse", sq_state, w_k)
        v = jnp.einsum("bsd,ed->bse", sq_state, w_v)
        attn = jax.nn.softmax(jnp.einsum("bhe,bse->bhs", q, k) * inv_sqrt_d, axis=-1)
        ctx = jnp.einsum("bhs,bse->bhe", attn, v)
        int_upd = layer_norm(int_h + jnp.einsum("bhe,de->bhd", ctx, w_o), params["b_rc_ln_g"], params["b_rc_ln_b"])
        hubs = jnp.concatenate([int_upd, hubs[:, n_int:, :]], axis=1)

        # Step B: Rich-Club Core 5.4x Exchange
        int_mean = jnp.mean(hubs[:, :n_int, :], axis=1)  # [B, D]
        mix_in = 0.5 * hubs + 0.5 * int_mean[:, None, :]
        rc_h = jax.nn.relu(jnp.einsum("bhd,ed->bhe", mix_in, w_rc1))
        rc_out = jnp.einsum("bhe,de->bhd", rc_h, w_rc2)
        hubs = layer_norm(hubs + rc_out, params["b_rc_ln_g"], params["b_rc_ln_b"])
        broad_mean = jnp.mean(hubs[:, n_int:, :], axis=1)

        # Step C: Central Complex Phasor Engine (PFN + vDelta - hDelta 180-deg roll - Delta7 - Ring)
        pfn_s = jnp.einsum("bd,rcd->brc", int_mean, w_pfn_s)
        pfn_o = jnp.einsum("bd,rcd->brc", broad_mean, w_pfn_o)
        v_step = jnp.einsum("brc,ec->bre", cx_phasor, w_vd)
        pfn_o_180 = jnp.roll(pfn_o, shift=sectors // 2, axis=1)  # Exact 180-degree phase shift!
        h_step = jnp.einsum("brc,ec->bre", pfn_o_180, w_hd)
        u_phasor = pfn_s + v_step - serotonin * h_step

        ring_in = jnp.concatenate([int_mean, mbon_summary], axis=-1)
        ring_gate = jax.nn.sigmoid(jnp.einsum("bi,ci->bc", ring_in, w_ring) + params["b_ring_b"])
        d7_scores = jnp.einsum("brc,ec->bre", u_phasor, w_d7)
        d7_ring = jax.nn.softmax(d7_scores, axis=1)
        cx_phasor = jax.nn.relu(u_phasor - d7_ring * jnp.abs(u_phasor) - 0.5 * ring_gate[:, None, :] * jnp.abs(u_phasor))
        cx_phasor = layer_norm(cx_phasor, params["b_cx_ln_g"], params["b_cx_ln_b"])

        # Step D: Visual Centrifugal Top-Down Inhibitory Gate (71.8% I -> 64 squares)
        cent_mask_64 = jax.nn.sigmoid(jnp.einsum("bd,sd->bs", broad_mean, w_cent) + params["b_cent_b"])
        gate_64 = 1.0 - 0.718 * (1.0 - cent_mask_64)[:, :, None]
        broad_proj = jnp.einsum("bd,ed->be", broad_mean, w_b2s)
        sq_in_upd = sq_state * gate_64 + 0.25 * broad_proj[:, None, :]
        sq_delta = jnp.tanh(jnp.einsum("bsd,ed->bse", sq_in_upd, w_squpd))
        sq_state = layer_norm(sq_state + 0.5 * sq_delta, params["b_sq_ln_g"], params["b_sq_ln_b"])

        # Deep Supervision tick WDL readout (fused with SEZ innate survival valence)
        tick_cx = jnp.mean(cx_phasor, axis=1)
        tick_global = jnp.concatenate([broad_mean, tick_cx], axis=-1)
        tick_wdl_raw = jnp.einsum("bg,kg->bk", tick_global, params["b_wdl_w"]) + params["b_wdl_b"]
        last_wdl_raw = tick_wdl_raw
        per_tick_wdl.append(tick_wdl_raw + sez_shift)

    cx_pooled = jnp.mean(cx_phasor, axis=1)
    global_vec = jnp.concatenate([broad_mean, cx_pooled], axis=-1)  # [B, D + cx_d]

    # B5. K = 4 Orthogonal Creative Goal Plan Bank
    plan_raw = jnp.tanh(jnp.einsum("bg,kcg->bkc", global_vec, params["b_plan_bank"]) + params["b_plan_b"][None, :, :])
    plan_norms = jnp.sqrt(jnp.sum(plan_raw ** 2, axis=-1, keepdims=True) + 1e-6)
    plans = plan_raw / plan_norms  # [B, 4, cx_d] unit vectors
    plan_scores = jnp.einsum("bkc,bc->bk", plans, cx_pooled)
    plan_w = jax.nn.softmax(plan_scores, axis=-1)
    committed_goal = jnp.einsum("bk,bkc->bc", plan_w, plans)

    # B6. Readout Heads: WDL, Direct CP, PFL2 Sharpness, Strategic Pressure, PFL3 64x64 Policy, Noduli, Seed
    wdl_logits = per_tick_wdl[-1]
    pos_probs_b = jax.nn.softmax(last_wdl_raw, axis=-1)
    deep_pos_cp = 400.0 * jnp.arctanh(jnp.clip(pos_probs_b[:, 0] - pos_probs_b[:, 2], -0.999, 0.999))
    pred_cp = jnp.clip(sez_vec + deep_pos_cp, -2500.0, 2500.0)
    pfl2_sharpness = jax.nn.sigmoid(jnp.einsum("bg,g->b", global_vec, params["b_pfl2_w"]) + params["b_pfl2_b"])
    strategic_pressure = jnp.tanh(jnp.einsum("bg,pg->bp", global_vec, params["b_press_w"]) + params["b_press_b"])

    # FC2 x EPG -> PFL3 Conjunctive Allocentric-to-Egocentric Goal Steering (Nature Feb 2024)
    conj_in = jnp.concatenate([committed_goal, cx_pooled], axis=-1)  # [B, 2 * cx_d]
    conj_steer = jnp.tanh(jnp.einsum("bi,ci->bc", conj_in, params["b_fc2_conj"]))
    eff_goal = committed_goal + 0.5 * conj_steer

    # PFL3 99% Contralateral Bilinear Policy Logits [B, 64, 64] -> flattened [B, 4096]
    q_sq = jnp.einsum("bsd,cd->bsc", sq_state, params["b_pfl3_q"])
    k_sq = jnp.einsum("bsd,cd->bsc", sq_state, params["b_pfl3_k"])
    o_sq = jnp.einsum("bsd,cd->bsc", sq_state, params["b_pfl3_opp"])
    aligned_q = q_sq * (1.0 + eff_goal[:, None, :])
    contra_k = k_sq - 0.99 * o_sq
    inv_sqrt_cx = float(1.0 / (cx_d ** 0.5))
    policy_logits = jnp.einsum("buc,bvc->buv", aligned_q, contra_k) * inv_sqrt_cx
    policy_flat = policy_logits.reshape((bsz, 4096))

    # Noduli Efference-Copy Counterfactual Causal Advantage Matrix [B, 64, 64] -> [B, 4096]
    nod_from = jnp.einsum("buc,c->bu", aligned_q, params["b_noduli_w"]) * inv_sqrt_cx  # [B, 64]
    nod_to = jnp.einsum("bvc,c->bv", contra_k, params["b_noduli_w"]) * inv_sqrt_cx     # [B, 64]
    noduli_delta_map = jnp.tanh(nod_from[:, :, None] - nod_to[:, None, :] + params["b_noduli_b"]).reshape((bsz, 4096))

    # Training-only AOTU -> LAL -> SMP 16-Causal-Concept Logits [B, 16] (with 25% head dropout)
    if "aux_causal_w" in params:
        if causal_drop_key is not None:
            keep_c = 0.75
            c_mask = (jax.random.uniform(causal_drop_key, global_vec.shape) < keep_c).astype(jnp.float32) / keep_c
            c_in = global_vec * c_mask
        else:
            c_in = global_vec
        causal_logits = jnp.einsum("bg,kg->bk", c_in, params["aux_causal_w"]) + params["aux_causal_b"]
    else:
        causal_logits = jnp.zeros((bsz, NUM_CAUSAL_CONCEPTS), dtype=jnp.float32)

    # Training-only 2-Ply VNC Causal Consequence Policy [B, 4096] (predicts follow-up move m_{t+2})
    if "aux_vnc2_q" in params:
        vq = jnp.einsum("bsd,cd->bsc", sq_state, params["aux_vnc2_q"]) * (1.0 + eff_goal[:, None, :])
        vk = jnp.einsum("bsd,cd->bsc", sq_state, params["aux_vnc2_k"])
        vnc2_logits = (jnp.einsum("buc,bvc->buv", vq, vk) * inv_sqrt_cx).reshape((bsz, 4096))
    else:
        vnc2_logits = policy_flat

    # Top-Down Stream B -> Stream A Seed & Centrifugal KC Mask
    seed_latent = jnp.tanh(jnp.einsum("bg,rg->br", global_vec, params["b_seed_down"]))
    gamma_seed = jnp.einsum("br,gr->bg", seed_latent, params["b_seed_to_gamma"])
    cent_kc_mask = jax.nn.sigmoid(jnp.einsum("br,ar->ba", seed_latent, params["b_seed_to_cent"]))

    plan_entropy = -jnp.mean(jnp.sum(plan_w * jnp.log(jnp.clip(plan_w, 1e-6, 1.0)), axis=-1))
    return {
        "pred_cp": pred_cp,
        "wdl_logits": wdl_logits,
        "per_tick_wdl": per_tick_wdl,
        "policy_logits": policy_flat,
        "noduli_delta_map": noduli_delta_map,
        "causal_logits": causal_logits,
        "vnc2_logits": vnc2_logits,
        "pfl2_sharpness": pfl2_sharpness,
        "strategic_pressure": strategic_pressure,
        "creative_plans": plans,
        "global_latent": global_vec,  # Used by Action-Conditioned hDelta World Model!
        "aligned_q": aligned_q,       # [B, 64, cx_d] used by hDelta World Model
        "contra_k": contra_k,         # [B, 64, cx_d] used by hDelta World Model
        "cent_board_mask": cent_mask_64,
        "gamma_seed": gamma_seed,
        "cent_kc_mask": cent_kc_mask,
        # Live brain-circuit health diagnostics
        "shunt_gain": shunt_gain,
        "lob_focus": lob_focus,
        "rc_hub_norm": jnp.mean(jnp.sqrt(jnp.sum(hubs ** 2, axis=-1) + 1e-6)),
        "cx_phasor_norm": jnp.mean(jnp.sqrt(jnp.sum(cx_phasor ** 2, axis=-1) + 1e-6)),
        "cent_focus": jnp.mean(jnp.std(cent_mask_64, axis=-1)),
        "plan_entropy": plan_entropy,
        "pfl2_mean": jnp.mean(pfl2_sharpness),
    }


def predict_action_world_model(
    params: Dict[str, jnp.ndarray],
    out_b: Dict[str, jnp.ndarray],
    move_u16: jnp.ndarray,  # [B] int32 packed (u << 6) | v
) -> jnp.ndarray:
    """Action-Conditioned hDelta Latent World Model: predicts future global_latent z_{t+2}
    conditioned on current latent z_t and the played action embedding do(m_t)!"""
    bsz = move_u16.shape[0]
    row = jnp.arange(bsz)
    u = (move_u16 >> 6) & 63
    v = move_u16 & 63
    act_emb = out_b["aligned_q"][row, u] - out_b["contra_k"][row, v]  # [B, cx_d]
    wm_in = jnp.concatenate([out_b["global_latent"], act_emb], axis=-1)
    h = jax.nn.silu(jnp.einsum("bi,hi->bh", wm_in, params["b_wm_w1"]) + params["b_wm_b1"])
    return out_b["global_latent"] + jnp.einsum("bh,oh->bo", h, params["b_wm_w2"])


def meta_dopamine_trajectory_scan(
    params: Dict[str, jnp.ndarray],
    cfg: SaccadionConfig,
    claws_idx: jnp.ndarray,
    traj_batch: Dict[str, jnp.ndarray],
    qat: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Saccadion-5 Differentiable 3-Lobe Meta-Plasticity over [B, K] correlated game trajectories.

    Unrolls BOTH ultra-plastic gamma (`dw_gamma_mat: [B, gamma, 8]`) AND semi-plastic
    DPM-consolidated alpha'/beta' (`dw_apbp_mat: [B, apbp, 8]`) injecting directly into the
    32-dim MBON vector via `a_plastic_to_mbon`! Trains the entire 3-lobe cognitive loop end-to-end!
    """
    bsz, _k_steps = traj_batch["phase_bucket"].shape
    _, gamma_dim, apbp_dim = cfg.lobe_dims_per_hemi()

    init_carry = (
        jnp.zeros((bsz, gamma_dim, PLASTIC_RANK), dtype=jnp.float32),  # dw_gamma_stm
        jnp.zeros((bsz, gamma_dim, PLASTIC_RANK), dtype=jnp.float32),  # dw_gamma_opp
        jnp.zeros((bsz, apbp_dim, PLASTIC_RANK), dtype=jnp.float32),   # dw_apbp_stm (DPM!)
        jnp.zeros((bsz, apbp_dim, PLASTIC_RANK), dtype=jnp.float32),   # dw_apbp_opp (DPM!)
        jnp.zeros((bsz, gamma_dim), dtype=jnp.float32),                # elig_stm
        jnp.zeros((bsz, gamma_dim), dtype=jnp.float32),                # elig_opp
        jnp.zeros((bsz, PLASTIC_RANK), dtype=jnp.float32),             # gamma_lesson_stm
        jnp.zeros((bsz,), dtype=jnp.float32),                          # prev_delta (Dop1R2)
        jnp.zeros((bsz,), dtype=jnp.float32),                          # prev_mbon_val (SOC)
    )

    eta_pam = jnp.clip(jax.nn.softplus(params["meta_eta_pam"]), 0.01, 0.5)
    eta_ppl1 = jnp.clip(jax.nn.softplus(params["meta_eta_ppl1"]), 0.02, 1.0)
    rho = jax.nn.sigmoid(params["meta_trace_decay"])
    eta_dop1r2 = jnp.clip(jax.nn.softplus(params.get("meta_eta_dop1r2", jnp.array(-1.9))), 0.01, 0.5)
    eta_soc = jnp.clip(jax.nn.softplus(params.get("meta_eta_soc", jnp.array(-1.4))), 0.01, 0.5)
    eta_dpm = jnp.clip(jax.nn.softplus(params.get("meta_eta_dpm", jnp.array(-1.8))), 0.01, 0.5)

    # Deterministic rank-8 channel polarity basis matching Rust plasticity.rs
    r_idx = jnp.arange(PLASTIC_RANK, dtype=jnp.float32)
    r_basis = jnp.where((r_idx % 2) == 0, 1.0, -0.5) / (1.0 + 0.25 * r_idx)  # [R]

    def scan_step(carry, step_inputs):
        dw_gs, dw_go, dw_as, dw_ao, elig_s, elig_o, g_lesson, prev_delta, prev_mbon_val = carry
        sp_s, sp_o, lh, ph, target_wdl = step_inputs

        out_a = forward_stream_a(
            params, cfg, claws_idx, sp_s, sp_o, lh, ph,
            dw_gamma_stm=dw_gs, dw_gamma_opp=dw_go,
            dw_apbp_stm=dw_as, dw_apbp_opp=dw_ao,
            qat=qat,
        )
        probs = jax.nn.softmax(out_a["wdl_logits"], axis=-1)
        pred_val = probs[:, 0] - probs[:, 2]
        true_val = target_wdl[:, 0] - target_wdl[:, 2]
        td_delta = jnp.clip(true_val - pred_val, -1.5, 1.5)

        # MBON -> DAN Second-Order Conditioning (SOC)
        mbon_val = jax.lax.stop_gradient(pred_val)
        soc_delta = jnp.clip(mbon_val - prev_mbon_val, -1.0, 1.0)
        delta = jnp.clip(td_delta + eta_soc * soc_delta, -1.5, 1.5)

        k_s = out_a["kc_gamma_stm"]
        k_o = out_a["kc_gamma_opp"]
        ap_s = out_a["kc_apbp_stm"]
        ap_o = out_a["kc_apbp_opp"]
        elig_s = rho * elig_s + k_s
        elig_o = rho * elig_o + k_o

        # Dop1R2 bidirectional causal order erosion on sign reversals
        reversal = jax.nn.sigmoid(-8.0 * (prev_delta * td_delta))
        dw_gs_e = dw_gs * (1.0 - eta_dop1r2 * reversal[:, None, None] * jnp.minimum(k_s[:, :, None] * 2.0, 1.0))
        dw_go_e = dw_go * (1.0 - eta_dop1r2 * reversal[:, None, None] * jnp.minimum(k_o[:, :, None] * 2.0, 1.0))

        norm_s = jnp.sum(k_s ** 2, axis=-1, keepdims=True) + 0.25 * jnp.sum(elig_s ** 2, axis=-1, keepdims=True) + 1e-4
        norm_o = jnp.sum(k_o ** 2, axis=-1, keepdims=True) + 0.25 * jnp.sum(elig_o ** 2, axis=-1, keepdims=True) + 1e-4
        norm_as = jnp.sum(ap_s ** 2, axis=-1, keepdims=True) + 1e-4
        norm_ao = jnp.sum(ap_o ** 2, axis=-1, keepdims=True) + 1e-4

        pos_d = jnp.maximum(delta, 0.0)[:, None]
        neg_d = jnp.maximum(-delta, 0.0)[:, None]
        # Asymmetric Dop1R1 PAM (+) vs 33.2% contralateral PPL1 (-) onto rank-8 gamma synapses
        g_drive_s = (eta_pam * pos_d - eta_ppl1 * neg_d) / norm_s  # [B, 1]
        g_drive_o = (0.332 * eta_ppl1 * neg_d) / norm_o            # [B, 1]
        upd_gs = g_drive_s[:, :, None] * (k_s + 0.5 * elig_s)[:, :, None] * r_basis[None, None, :]
        upd_go = g_drive_o[:, :, None] * (k_o + 0.5 * elig_o)[:, :, None] * r_basis[None, None, :]
        dw_gs_next = jnp.clip(0.98 * dw_gs_e + upd_gs, -1.5, 1.5)
        dw_go_next = jnp.clip(0.98 * dw_go_e + upd_go, -1.5, 1.5)

        # True DPM Cross-Lobe Consolidation: slowly transfers gamma synaptic lesson -> semi-plastic alpha'/beta'!
        step_lesson = jnp.mean(upd_gs, axis=1) * 4.0  # [B, R]
        g_lesson_next = jnp.clip(0.85 * g_lesson + 0.15 * step_lesson, -1.5, 1.5)
        dpm_sig_s = g_lesson_next + 0.35 * delta[:, None] * r_basis[None, :]
        dpm_sig_o = -0.15 * delta[:, None] * r_basis[None, :]
        dw_as_next = jnp.clip(
            0.96 * dw_as + (eta_dpm * ap_s[:, :, None] / norm_as[:, :, None]) * dpm_sig_s[:, None, :],
            -1.2, 1.2,
        )
        dw_ao_next = jnp.clip(
            0.96 * dw_ao + (eta_dpm * ap_o[:, :, None] / norm_ao[:, :, None]) * dpm_sig_o[:, None, :],
            -1.2, 1.2,
        )
        return (
            dw_gs_next, dw_go_next, dw_as_next, dw_ao_next,
            elig_s, elig_o, g_lesson_next, td_delta, mbon_val,
        ), (out_a["wdl_logits"], out_a["active_kc_ratio"])

    seq_in = (
        jnp.swapaxes(traj_batch["sparse_stm"], 0, 1),
        jnp.swapaxes(traj_batch["sparse_opp"], 0, 1),
        jnp.swapaxes(traj_batch["lh_feat"], 0, 1),
        jnp.swapaxes(traj_batch["phase_bucket"], 0, 1),
        jnp.swapaxes(traj_batch["wdl_target"], 0, 1),
    )
    _, (all_logits, all_ratios) = jax.lax.scan(scan_step, init_carry, seq_in)
    return {
        "traj_wdl_logits": jnp.swapaxes(all_logits, 0, 1),  # [B, K, 3]
        "active_kc_ratio": jnp.mean(all_ratios),
    }
