"""Exact `.saccnet` binary exporter and Pure-NumPy / Rust bit-exact parity verifier.

Writes tensors in the EXACT order expected by `crates/saccadion-nn/src/weights.rs::load_saccnet`.
Also includes a dependency-free NumPy reference forward pass so Python <-> Rust equivalence
can be verified on any CPU machine even before uploading to Colab TPU!
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from .config import (
    OPTIC_LAYERS,
    SACCNET_MAGIC,
    SACCNET_VERSION,
    SaccadionConfig,
    build_flyhash_claws,
)

# Canonical tensor serialization order matching Rust `SaccadionWeights::load_saccnet`
SACCNET_TENSOR_ORDER = [
    "a_ft_weight",
    "a_ft_bias",
    "a_sez_weight",
    "a_claw_weight",
    "a_apl_beta",
    "a_apl_thresh",
    "a_mbon_ipsi",
    "a_mbon_contra",
    "a_mbon_bias",
    "a_lh_w_inh",
    "a_lh_b_inh",
    "a_lh_w_out",
    "a_lh_b_out",
    "a_smp_w",
    "a_smp_b",
    "b_shunt_exc",
    "b_shunt_inh",
    "b_shunt_leak",
    "b_lc_w",
    "b_optic_b",
    "b_hub_init",
    "b_int_q",
    "b_int_k",
    "b_int_v",
    "b_int_o",
    "b_rc_mix1",
    "b_rc_mix2",
    "b_rc_ln_g",
    "b_rc_ln_b",
    "b_pfn_stm",
    "b_pfn_opp",
    "b_vdelta",
    "b_hdelta",
    "b_delta7",
    "b_ring_inh",
    "b_ring_b",
    "b_cent_w",
    "b_cent_b",
    "b_broad_to_sq",
    "b_sq_upd",
    "b_sq_ln_g",
    "b_sq_ln_b",
    "b_plan_bank",
    "b_plan_b",
    "b_pfl3_q",
    "b_pfl3_k",
    "b_pfl3_opp",
    "b_pfl2_w",
    "b_pfl2_b",
    "b_wdl_w",
    "b_wdl_b",
    "b_press_w",
    "b_press_b",
    "b_noduli_w",
    "b_noduli_b",
    "b_seed_down",
    "b_seed_to_gamma",
    "b_seed_to_cent",
    "b_opt_ln1_g",
    "b_opt_ln1_b",
    "b_opt_q",
    "b_opt_k",
    "b_opt_v",
    "b_opt_o",
    "b_opt_rel",
    "b_opt_ln2_g",
    "b_opt_ln2_b",
    "b_opt_fg",
    "b_opt_fu",
    "b_opt_fd",
]


def export_params_to_saccnet(
    cfg: SaccadionConfig, params: Dict[str, np.ndarray], out_path: Path
) -> None:
    """Exports a JAX or NumPy parameter dict directly to `.saccnet` readable by Pure Rust."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<8s18IQ",
        SACCNET_MAGIC,
        SACCNET_VERSION,
        cfg.tier_id,
        cfg.sparse_input_dim,
        cfg.acc_dim,
        cfg.flyhash_claws,
        cfg.mbon_dim,
        cfg.num_phase_stacks,
        cfg.lh_input_dim,
        cfg.lh_hidden_dim,
        cfg.sq_feat_dim,
        cfg.stream_b_dim,
        cfg.num_integrators,
        cfg.num_broadcasters,
        cfg.cx_sectors,
        cfg.cx_dim,
        cfg.cx_max_ticks,
        cfg.num_creative_plans,
        cfg.seed_rank,
        cfg.flyhash_seed,
    )
    with open(out_path, "wb") as f:
        f.write(header)
        for name in SACCNET_TENSOR_ORDER:
            arr = np.ascontiguousarray(np.asarray(params[name], dtype="<f4"))
            f.write(arr.tobytes())


def init_numpy_params(cfg: SaccadionConfig, seed: int = 1234) -> Dict[str, np.ndarray]:
    """Pure-NumPy initializer with identical shapes to JAX `init_saccadion_params`."""
    from .config import init_sez_innate_weights

    rng = np.random.default_rng(seed)
    u = lambda shape, scale: rng.uniform(-scale, scale, size=shape).astype(np.float32)

    _, gamma_dim, _ = cfg.lobe_dims_per_hemi()
    plastic_dim = cfg.plastic_dim_per_hemi()
    d, cx, mbon, stacks = cfg.stream_b_dim, cfg.cx_dim, cfg.mbon_dim, cfg.num_phase_stacks
    smp_in = 2 * mbon + 4
    s_d = (1.0 / d) ** 0.5 * 0.4
    s_cx = (1.0 / cx) ** 0.5 * 0.4

    return {
        "a_ft_weight": u((cfg.sparse_input_dim, cfg.acc_dim), 0.05),
        "a_ft_bias": np.full((cfg.acc_dim,), 0.05, dtype=np.float32),
        "a_sez_weight": init_sez_innate_weights(cfg.sparse_input_dim, stacks),
        "a_claw_weight": u((plastic_dim, cfg.flyhash_claws), 0.35),
        "a_apl_beta": np.array([0.85, 0.85], dtype=np.float32),
        "a_apl_thresh": np.full((cfg.acc_dim,), 0.18, dtype=np.float32),
        "a_mbon_ipsi": u((stacks, cfg.acc_dim, mbon), 0.1),
        "a_mbon_contra": u((stacks, cfg.acc_dim, mbon), 0.1),
        "a_mbon_bias": np.zeros((stacks, mbon), dtype=np.float32),
        "a_lh_w_inh": u((cfg.lh_hidden_dim, cfg.lh_input_dim), 0.15),
        "a_lh_b_inh": np.full((cfg.lh_hidden_dim,), 0.5, dtype=np.float32),
        "a_lh_w_out": u((4, cfg.lh_hidden_dim), 0.2),
        "a_lh_b_out": np.array([-1.2, -1.5, -1.5, -1.2], dtype=np.float32),
        "a_smp_w": u((stacks, 3, smp_in), 0.1),
        "a_smp_b": np.zeros((stacks, 3), dtype=np.float32),
        "b_shunt_exc": u((d, cfg.sq_feat_dim), s_d),
        "b_shunt_inh": u((d, cfg.sq_feat_dim), s_d),
        "b_shunt_leak": np.full((d,), 0.6, dtype=np.float32),
        "b_lc_w": u((d, cfg.sq_feat_dim), s_d),
        "b_optic_b": np.zeros((d,), dtype=np.float32),
        "b_hub_init": u((cfg.num_integrators + cfg.num_broadcasters, d), s_d),
        "b_int_q": u((d, d), s_d),
        "b_int_k": u((d, d), s_d),
        "b_int_v": u((d, d), s_d),
        "b_int_o": u((d, d), s_d),
        "b_rc_mix1": u((d, d), s_d),
        "b_rc_mix2": u((d, d), s_d),
        "b_rc_ln_g": np.ones((d,), dtype=np.float32),
        "b_rc_ln_b": np.zeros((d,), dtype=np.float32),
        "b_pfn_stm": u((cfg.cx_sectors, cx, d), s_d),
        "b_pfn_opp": u((cfg.cx_sectors, cx, d), s_d),
        "b_vdelta": u((cx, cx), s_cx),
        "b_hdelta": u((cx, cx), s_cx),
        "b_delta7": u((cx, cx), s_cx),
        "b_ring_inh": u((cx, d + mbon), s_d),
        "b_ring_b": np.zeros((cx,), dtype=np.float32),
        "b_cent_w": u((64, d), s_d),
        "b_cent_b": np.full((64,), 0.5, dtype=np.float32),
        "b_broad_to_sq": u((d, d), s_d),
        "b_sq_upd": u((d, d), s_d),
        "b_sq_ln_g": np.ones((d,), dtype=np.float32),
        "b_sq_ln_b": np.zeros((d,), dtype=np.float32),
        "b_plan_bank": u((cfg.num_creative_plans, cx, cx + d), s_d),
        "b_plan_b": np.zeros((cfg.num_creative_plans, cx), dtype=np.float32),
        "b_pfl3_q": u((cx, d), s_d),
        "b_pfl3_k": u((cx, d), s_d),
        "b_pfl3_opp": u((cx, d), s_d),
        "b_pfl2_w": u((d + cx,), s_d),
        "b_pfl2_b": np.array(0.0, dtype=np.float32),
        "b_wdl_w": u((3, d + cx), s_d),
        "b_wdl_b": np.zeros((3,), dtype=np.float32),
        "b_press_w": u((5, d + cx), s_d),
        "b_press_b": np.zeros((5,), dtype=np.float32),
        "b_noduli_w": u((cx,), s_cx),
        "b_noduli_b": np.array(0.0, dtype=np.float32),
        "b_seed_down": u((cfg.seed_rank, d + cx), s_d),
        "b_seed_to_gamma": u((gamma_dim, cfg.seed_rank), 0.1),
        "b_seed_to_cent": u((cfg.acc_dim, cfg.seed_rank), 0.1),
        "b_opt_ln1_g": np.ones((OPTIC_LAYERS, d), dtype=np.float32),
        "b_opt_ln1_b": np.zeros((OPTIC_LAYERS, d), dtype=np.float32),
        "b_opt_q": u((OPTIC_LAYERS, d, d), s_d),
        "b_opt_k": u((OPTIC_LAYERS, d, d), s_d),
        "b_opt_v": u((OPTIC_LAYERS, d, d), s_d),
        "b_opt_o": u((OPTIC_LAYERS, d, d), s_d * 0.5),
        "b_opt_rel": np.zeros((OPTIC_LAYERS, 4, 64, 64), dtype=np.float32),
        "b_opt_ln2_g": np.ones((OPTIC_LAYERS, d), dtype=np.float32),
        "b_opt_ln2_b": np.zeros((OPTIC_LAYERS, d), dtype=np.float32),
        "b_opt_fg": u((OPTIC_LAYERS, 2 * d, d), s_d),
        "b_opt_fu": u((OPTIC_LAYERS, 2 * d, d), s_d),
        "b_opt_fd": u((OPTIC_LAYERS, d, 2 * d), s_d * 0.5),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _ln(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5) * g + b


def verify_python_to_rust_parity(rust_binary: Path, tmp_saccnet: Path) -> None:
    """Exports Tiny weights from Python -> `.saccnet`, runs Rust `saccadion parity-check`,
    and verifies exact Startpos numerical agreement across Stream A and Stream B!"""
    from .dataset import SACC_RECORD_DTYPE, decode_record_features

    cfg = SaccadionConfig.tiny()
    params = init_numpy_params(cfg, seed=2026)
    export_params_to_saccnet(cfg, params, tmp_saccnet)

    # Construct exact startpos packed SaccRecord in Python
    start_pieces = [
        4, 2, 3, 5, 6, 3, 2, 4,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        7, 7, 7, 7, 7, 7, 7, 7,
        10, 8, 9, 11, 12, 9, 8, 10,
    ]
    rec = np.zeros((), dtype=SACC_RECORD_DTYPE)
    for sq, code in enumerate(start_pieces):
        if (sq & 1) == 0:
            rec["board_nibbles"][sq >> 1] |= code
        else:
            rec["board_nibbles"][sq >> 1] |= code << 4
    # stm=0, castling=0xF (<<1), ep=0, halfmove=0, phase=7 (30 non-king / 4 -> min 7)
    rec["meta_flags"] = (0x0F << 1) | (7 << 17)
    sp_s, sp_o, board, lh_feat, phase = decode_record_features(rec, max_sparse=96)

    # Stream A reference forward pass (NumPy mirror of model.py + SEZ innate valence)
    ab_dim, _, _ = cfg.lobe_dims_per_hemi()
    claws = build_flyhash_claws(cfg)
    acc_s = np.sum(params["a_ft_weight"][sp_s[sp_s >= 0]], axis=0) + params["a_ft_bias"]
    acc_o = np.sum(params["a_ft_weight"][sp_o[sp_o >= 0]], axis=0) + params["a_ft_bias"]
    sez_s = float(np.sum(params["a_sez_weight"][sp_s[sp_s >= 0], phase]))
    sez_o = float(np.sum(params["a_sez_weight"][sp_o[sp_o >= 0], phase]))
    sez_cp_py = (sez_s - sez_o) * 50.0
    pre_s = np.concatenate([acc_s[:ab_dim], np.sum(acc_s[claws] * params["a_claw_weight"], axis=-1)])
    pre_o = np.concatenate([acc_o[:ab_dim], np.sum(acc_o[claws] * params["a_claw_weight"], axis=-1)])
    kc_s = np.clip(pre_s - params["a_apl_beta"][0] * np.mean(np.maximum(pre_s, 0.0)) - params["a_apl_thresh"], 0.0, 1.0) ** 2
    kc_o = np.clip(pre_o - params["a_apl_beta"][1] * np.mean(np.maximum(pre_o, 0.0)) - params["a_apl_thresh"], 0.0, 1.0) ** 2

    mbon_s = kc_s @ params["a_mbon_ipsi"][phase] + 0.5 * (kc_o @ params["a_mbon_contra"][phase]) + params["a_mbon_bias"][phase]
    mbon_o = kc_o @ params["a_mbon_ipsi"][phase] + 0.5 * (kc_s @ params["a_mbon_contra"][phase]) + params["a_mbon_bias"][phase]
    signs = np.concatenate([np.ones(cfg.mbon_dim // 2), -np.ones(cfg.mbon_dim // 2)])
    mbon_s = _silu(mbon_s) * signs
    mbon_o = _silu(mbon_o) * signs

    lh_h = np.maximum(params["a_lh_b_inh"] - (np.abs(params["a_lh_w_inh"]) @ lh_feat), 0.0)
    lh_raw = _sigmoid(params["a_lh_w_out"] @ lh_h + params["a_lh_b_out"])
    lh_veto = np.array([0.7 * lh_raw[0], 0.65 * lh_raw[1], 0.7 * lh_raw[2], 0.7 * lh_raw[3]], dtype=np.float32)

    smp_in = np.concatenate([mbon_s, mbon_o, lh_veto])
    wdl_a_raw = params["a_smp_w"][phase] @ smp_in + params["a_smp_b"][phase]
    mbon_probs_a = _softmax(wdl_a_raw)
    exp_a = float(np.clip(mbon_probs_a[0] - mbon_probs_a[2], -0.999, 0.999))
    mbon_cp_py = float(np.clip(400.0 * np.arctanh(exp_a), -2500.0, 2500.0))
    slow_cp_py = float(np.clip((sez_cp_py + mbon_cp_py) * (1.0 - 0.55 * lh_veto[1]), -2500.0, 2500.0))
    mbon_summary = mbon_s - 0.5 * mbon_o

    # Stream B reference forward pass (all cx_max_ticks, adaptive_halt=False)
    d, cx_d, n_int, sectors = cfg.stream_b_dim, cfg.cx_dim, cfg.num_integrators, cfg.cx_sectors
    exc = np.maximum(board @ params["b_shunt_exc"].T, 0.0)
    inh = np.maximum(board @ params["b_shunt_inh"].T, 0.0)
    leak = np.log1p(np.exp(params["b_shunt_leak"])) + 0.5
    sq_state = exc / (leak + inh) + board @ params["b_lc_w"].T + params["b_optic_b"]
    hubs = params["b_hub_init"].copy()
    cx_p = np.zeros((sectors, cx_d), dtype=np.float32)
    inv_sd = 1.0 / np.sqrt(d)

    n_lob_h = 4
    lob_hd = d // n_lob_h
    inv_shd = 1.0 / np.sqrt(lob_hd)

    # B1b. 3-Layer Untied FlyWire Optic Lobe Stack (Dm/Pm Attention + Tm/TmY SwiGLU FFN)
    for l_idx in range(OPTIC_LAYERS):
        x1 = _ln(sq_state, params["b_opt_ln1_g"][l_idx], params["b_opt_ln1_b"][l_idx])
        lq = (x1 @ params["b_opt_q"][l_idx].T).reshape(64, n_lob_h, lob_hd)
        lk = (x1 @ params["b_opt_k"][l_idx].T).reshape(64, n_lob_h, lob_hd)
        lv = (x1 @ params["b_opt_v"][l_idx].T).reshape(64, n_lob_h, lob_hd)
        l_scores = np.einsum("uhd,vhd->huv", lq, lk) * inv_shd + params["b_opt_rel"][l_idx]
        l_attn = _softmax(l_scores, axis=-1)
        l_ctx = np.einsum("huv,vhd->uhd", l_attn, lv).reshape(64, d)
        sq_state = sq_state + l_ctx @ params["b_opt_o"][l_idx].T

        x2 = _ln(sq_state, params["b_opt_ln2_g"][l_idx], params["b_opt_ln2_b"][l_idx])
        fg = _silu(x2 @ params["b_opt_fg"][l_idx].T)
        fu = x2 @ params["b_opt_fu"][l_idx].T
        sq_state = sq_state + (fg * fu) @ params["b_opt_fd"][l_idx].T

    for _ in range(cfg.cx_max_ticks):
        int_h = hubs[:n_int]
        q = int_h @ params["b_int_q"].T
        k = sq_state @ params["b_int_k"].T
        v = sq_state @ params["b_int_v"].T
        attn = _softmax((q @ k.T) * inv_sd, axis=-1)
        ctx = attn @ v
        hubs[:n_int] = _ln(int_h + ctx @ params["b_int_o"].T, params["b_rc_ln_g"], params["b_rc_ln_b"])

        int_mean = np.mean(hubs[:n_int], axis=0)
        mix_in = 0.5 * hubs + 0.5 * int_mean[None, :]
        rc_h = np.maximum(mix_in @ params["b_rc_mix1"].T, 0.0)
        hubs = _ln(hubs + rc_h @ params["b_rc_mix2"].T, params["b_rc_ln_g"], params["b_rc_ln_b"])
        broad_mean = np.mean(hubs[n_int:], axis=0)

        pfn_s = np.einsum("rcd,d->rc", params["b_pfn_stm"], int_mean)
        pfn_o = np.einsum("rcd,d->rc", params["b_pfn_opp"], broad_mean)
        v_step = cx_p @ params["b_vdelta"].T
        h_step = np.roll(pfn_o, shift=sectors // 2, axis=0) @ params["b_hdelta"].T
        u_p = pfn_s + v_step - h_step
        ring_gate = _sigmoid(params["b_ring_inh"] @ np.concatenate([int_mean, mbon_summary]) + params["b_ring_b"])
        d7_ring = _softmax(u_p @ params["b_delta7"].T, axis=0)
        cx_p = np.maximum(u_p - d7_ring * np.abs(u_p) - 0.5 * ring_gate[None, :] * np.abs(u_p), 0.0)

        cent_64 = _sigmoid(params["b_cent_w"] @ broad_mean + params["b_cent_b"])
        gate_64 = 1.0 - 0.718 * (1.0 - cent_64)[:, None]
        broad_proj = params["b_broad_to_sq"] @ broad_mean
        sq_upd = np.tanh((sq_state * gate_64 + 0.25 * broad_proj[None, :]) @ params["b_sq_upd"].T)
        sq_state = _ln(sq_state + 0.5 * sq_upd, params["b_sq_ln_g"], params["b_sq_ln_b"])

    global_vec = np.concatenate([broad_mean, np.mean(cx_p, axis=0)])
    wdl_b_raw = params["b_wdl_w"] @ global_vec + params["b_wdl_b"]
    pos_probs_b = _softmax(wdl_b_raw)
    deep_pos_cp = float(np.clip(400.0 * np.arctanh(np.clip(pos_probs_b[0] - pos_probs_b[2], -0.999, 0.999)), -2500.0, 2500.0))
    deep_cp_py = float(np.clip(sez_cp_py + deep_pos_cp, -2500.0, 2500.0))
    pfl2_py = float(_sigmoid(np.dot(params["b_pfl2_w"], global_vec) + params["b_pfl2_b"]))
    press_py = np.tanh(params["b_press_w"] @ global_vec + params["b_press_b"])

    # Invoke Pure-Rust binary on the exported `.saccnet` file!
    res = subprocess.run(
        [str(rust_binary), "parity-check", str(tmp_saccnet)],
        capture_output=True,
        text=True,
        check=True,
    )
    line = res.stdout.strip()
    print("Rust output:", line)
    kv = dict(tok.split("=") for tok in line.split()[1:])
    rust_a_cp = float(kv["stream_a_slow_cp"])
    rust_b_cp = float(kv["stream_b_deep_cp"])
    rust_pfl2 = float(kv["pfl2"])
    rust_press = np.array([float(x) for x in kv["press"].split(",")], dtype=np.float32)

    print(f"Stream A slow_cp: Python={slow_cp_py:.6f} | Rust={rust_a_cp:.6f} | diff={abs(slow_cp_py - rust_a_cp):.2e}")
    print(f"Stream B deep_cp: Python={deep_cp_py:.6f} | Rust={rust_b_cp:.6f} | diff={abs(deep_cp_py - rust_b_cp):.2e}")
    print(f"Stream B PFL2   : Python={pfl2_py:.6f} | Rust={rust_pfl2:.6f} | diff={abs(pfl2_py - rust_pfl2):.2e}")
    print(f"Pressure maxdiff: {np.max(np.abs(press_py - rust_press)):.2e}")

    assert abs(slow_cp_py - rust_a_cp) < 1e-4, "Stream A cp parity mismatch"
    assert abs(deep_cp_py - rust_b_cp) < 1e-4, "Stream B cp parity mismatch"
    assert abs(pfl2_py - rust_pfl2) < 1e-5, "PFL2 parity mismatch"
    assert np.max(np.abs(press_py - rust_press)) < 1e-5, "Strategic Pressure parity mismatch"
    print("EXACT PYTHON <-> RUST .SACCNET PARITY VERIFIED TO < 1e-4!")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    rust_bin = root / "target" / "release" / "saccadion.exe"
    tmp_net = root / "training" / "_parity_test.saccnet"
    try:
        verify_python_to_rust_parity(rust_bin, tmp_net)
    finally:
        if tmp_net.exists():
            tmp_net.unlink()
