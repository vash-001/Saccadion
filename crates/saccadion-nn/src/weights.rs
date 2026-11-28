//! Complete Saccadion-1 weight storage and deterministic `.saccnet` binary serialization.
//!
//! Layout is shared identically with `training/saccadion_tpu/export_rust.py` so JAX-trained
//! checkpoints load directly into Pure Rust with bit-verified numerical parity.

use crate::config::{
    build_flyhash_claws, splitmix64, ModelConfig, AMACRINE_RANK, GF_H1, GF_H2, GRAPH_EDGE_TYPES,
    OPTIC_LAYERS, PLASTIC_RANK, SACCNET_MAGIC, SACCNET_VERSION,
};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Quantization scale for AVX2 INT16 HalfKA accumulator (128 units = 1.0 float -> +/-256.0 range).
pub const FT_I16_SCALE: f32 = 128.0;
pub const FT_I16_INV_SCALE: f32 = 1.0 / FT_I16_SCALE;

fn quantize_to_i16(src: &[f32]) -> Vec<i16> {
    src.iter()
        .map(|&v| (v * FT_I16_SCALE).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16)
        .collect()
}

/// Packs only the frozen alpha/beta lobe (`0..ab_dim`) of every sparse feature row contiguously
/// in INT16 (2.7x smaller L2/L3 cache footprint for blazing Push/Pop Incremental Giant Fiber!).
fn pack_ab_lobe_i16(full_i16: &[i16], num_rows: usize, full_dim: usize, ab_dim: usize) -> Vec<i16> {
    let mut out = vec![0i16; num_rows * ab_dim];
    for r in 0..num_rows {
        let src = &full_i16[r * full_dim..r * full_dim + ab_dim];
        let dst = &mut out[r * ab_dim..(r + 1) * ab_dim];
        dst.copy_from_slice(src);
    }
    out
}

fn quantize_stacks_to_i8(src: &[f32], num_stacks: usize) -> (Vec<i8>, Vec<f32>) {
    let per_stack = src.len() / num_stacks.max(1);
    let mut out = vec![0i8; src.len()];
    let mut scales = vec![1.0f32; num_stacks];
    for s in 0..num_stacks {
        let sl = &src[s * per_stack..(s + 1) * per_stack];
        let max_abs = sl.iter().fold(1e-6f32, |m, &v| m.max(v.abs()));
        let scale = max_abs / 127.0;
        let inv = 127.0 / max_abs;
        scales[s] = scale;
        let dst = &mut out[s * per_stack..(s + 1) * per_stack];
        for i in 0..per_stack {
            dst[i] = (sl[i] * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (out, scales)
}

/// Per-row symmetrically quantized INT8 matrix for fast L2-resident AVX2 Stream B matvecs.
#[derive(Clone, Debug)]
pub struct QuantizedMatrixI8 {
    pub data: Vec<i8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl QuantizedMatrixI8 {
    pub fn from_f32(src: &[f32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(src.len(), rows * cols);
        let mut data = vec![0i8; rows * cols];
        let mut scales = vec![1.0f32; rows];
        for r in 0..rows {
            let row = &src[r * cols..(r + 1) * cols];
            let max_abs = row.iter().fold(1e-6f32, |m, &v| m.max(v.abs()));
            scales[r] = max_abs / 127.0;
            let inv = 127.0 / max_abs;
            let dst = &mut data[r * cols..(r + 1) * cols];
            for c in 0..cols {
                dst[c] = (row[c] * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
        Self { data, scales, rows, cols }
    }
}

/// Pre-quantized runtime INT8 cache for Stream B's large dense matrices (4x smaller L2 footprint).
#[derive(Clone, Debug)]
pub struct StreamBQuantCache {
    pub gnn_w: Vec<QuantizedMatrixI8>,
    pub retro_w: Vec<QuantizedMatrixI8>,
    pub opt_q: Vec<QuantizedMatrixI8>,
    pub opt_k: Vec<QuantizedMatrixI8>,
    pub opt_v: Vec<QuantizedMatrixI8>,
    pub opt_o: Vec<QuantizedMatrixI8>,
    pub opt_fg: Vec<QuantizedMatrixI8>,
    pub opt_fu: Vec<QuantizedMatrixI8>,
    pub opt_fd: Vec<QuantizedMatrixI8>,
    pub int_q: QuantizedMatrixI8,
    pub int_k: QuantizedMatrixI8,
    pub int_v: QuantizedMatrixI8,
    pub int_o: QuantizedMatrixI8,
    pub rc_mix1: QuantizedMatrixI8,
    pub rc_mix2: QuantizedMatrixI8,
    pub broad_to_sq: QuantizedMatrixI8,
    pub sq_upd: QuantizedMatrixI8,
    pub wm_w1: QuantizedMatrixI8,
    pub wm_w2: QuantizedMatrixI8,
}

impl StreamBQuantCache {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        d: usize,
        cx: usize,
        b_gnn_w: &[f32],
        b_retro_w: &[f32],
        b_opt_q: &[f32],
        b_opt_k: &[f32],
        b_opt_v: &[f32],
        b_opt_o: &[f32],
        b_opt_fg: &[f32],
        b_opt_fu: &[f32],
        b_opt_fd: &[f32],
        b_int_q: &[f32],
        b_int_k: &[f32],
        b_int_v: &[f32],
        b_int_o: &[f32],
        b_rc_mix1: &[f32],
        b_rc_mix2: &[f32],
        b_broad_to_sq: &[f32],
        b_sq_upd: &[f32],
        b_wm_w1: &[f32],
        b_wm_w2: &[f32],
    ) -> Self {
        let gnn_w = (0..GRAPH_EDGE_TYPES)
            .map(|k| QuantizedMatrixI8::from_f32(&b_gnn_w[k * d * d..(k + 1) * d * d], d, d))
            .collect();
        let retro_w = (0..GRAPH_EDGE_TYPES)
            .map(|k| QuantizedMatrixI8::from_f32(&b_retro_w[k * d * d..(k + 1) * d * d], d, d))
            .collect();
        let ffn_d = 2 * d;
        let opt_q = (0..OPTIC_LAYERS)
            .map(|l| QuantizedMatrixI8::from_f32(&b_opt_q[l * d * d..(l + 1) * d * d], d, d))
            .collect();
        let opt_k = (0..OPTIC_LAYERS)
            .map(|l| QuantizedMatrixI8::from_f32(&b_opt_k[l * d * d..(l + 1) * d * d], d, d))
            .collect();
        let opt_v = (0..OPTIC_LAYERS)
            .map(|l| QuantizedMatrixI8::from_f32(&b_opt_v[l * d * d..(l + 1) * d * d], d, d))
            .collect();
        let opt_o = (0..OPTIC_LAYERS)
            .map(|l| QuantizedMatrixI8::from_f32(&b_opt_o[l * d * d..(l + 1) * d * d], d, d))
            .collect();
        let opt_fg = (0..OPTIC_LAYERS)
            .map(|l| QuantizedMatrixI8::from_f32(&b_opt_fg[l * ffn_d * d..(l + 1) * ffn_d * d], ffn_d, d))
            .collect();
        let opt_fu = (0..OPTIC_LAYERS)
            .map(|l| QuantizedMatrixI8::from_f32(&b_opt_fu[l * ffn_d * d..(l + 1) * ffn_d * d], ffn_d, d))
            .collect();
        let opt_fd = (0..OPTIC_LAYERS)
            .map(|l| QuantizedMatrixI8::from_f32(&b_opt_fd[l * d * ffn_d..(l + 1) * d * ffn_d], d, ffn_d))
            .collect();
        Self {
            gnn_w,
            retro_w,
            opt_q,
            opt_k,
            opt_v,
            opt_o,
            opt_fg,
            opt_fu,
            opt_fd,
            int_q: QuantizedMatrixI8::from_f32(b_int_q, d, d),
            int_k: QuantizedMatrixI8::from_f32(b_int_k, d, d),
            int_v: QuantizedMatrixI8::from_f32(b_int_v, d, d),
            int_o: QuantizedMatrixI8::from_f32(b_int_o, d, d),
            rc_mix1: QuantizedMatrixI8::from_f32(b_rc_mix1, d, d),
            rc_mix2: QuantizedMatrixI8::from_f32(b_rc_mix2, d, d),
            broad_to_sq: QuantizedMatrixI8::from_f32(b_broad_to_sq, d, d),
            sq_upd: QuantizedMatrixI8::from_f32(b_sq_upd, d, d),
            wm_w1: QuantizedMatrixI8::from_f32(b_wm_w1, d + cx, d + 2 * cx),
            wm_w2: QuantizedMatrixI8::from_f32(b_wm_w2, d + cx, d + cx),
        }
    }
}

/// All static (frozen after TPU training) weights of Stream A + Stream B.
#[derive(Clone, Debug)]
pub struct SaccadionWeights {
    pub cfg: ModelConfig,
    /// Deterministic FlyHash sparse claw indices: [plastic_dim_per_hemi * flyhash_claws].
    pub flyhash_claws: Vec<u16>,

    // ==================== STREAM A (Fast Reflex, SEZ & Calyx) ====================
    /// Sparse feature transformer: [sparse_input_dim * acc_dim] (row-major by feature index).
    pub a_ft_weight: Vec<f32>,
    pub a_ft_bias: Vec<f32>,
    /// SEZ (Subesophageal Zone) Innate Survival Valence table: [sparse_input_dim * num_phase_stacks].
    /// Initialized to biological piece-square survival values (P=1, N=3.2, B=3.3, R=5, Q=9 pawns)
    /// and fine-tuned end-to-end so MBONs/CX modulate innate value instead of re-learning material!
    pub a_sez_weight: Vec<f32>,
    /// FlyHash claw weights for plastic KCs: [plastic_dim_per_hemi * flyhash_claws].
    pub a_claw_weight: Vec<f32>,
    /// APL global GABA inhibition scaling per hemisphere [2] and per-KC thresholds [acc_dim].
    pub a_apl_beta: [f32; 2],
    pub a_apl_thresh: Vec<f32>,
    /// Push-Pull MBON projections (2:1 ipsi/contra) per game-phase stack:
    /// [num_phase_stacks * acc_dim * mbon_dim] for ipsi and contra.
    pub a_mbon_ipsi: Vec<f32>,
    pub a_mbon_contra: Vec<f32>,
    pub a_mbon_bias: Vec<f32>, // [num_phase_stacks * mbon_dim]
    /// Lateral Horn 98.2%-inhibitory innate veto branch:
    /// w_inh: [lh_hidden_dim * lh_input_dim], b_inh: [lh_hidden_dim],
    /// w_out: [4 * lh_hidden_dim], b_out: [4].
    pub a_lh_w_inh: Vec<f32>,
    pub a_lh_b_inh: Vec<f32>,
    pub a_lh_w_out: Vec<f32>,
    pub a_lh_b_out: [f32; 4],
    /// SMP/CRE/SIP Convergence Head per stack -> 3 WDL logits + scalar cp calibration:
    /// Input is [mbon_stm (32) || mbon_opp (32) || lh_veto (4)] = 68.
    pub a_smp_w: Vec<f32>, // [num_phase_stacks * 3 * (2 * mbon_dim + 4)]
    pub a_smp_b: Vec<f32>, // [num_phase_stacks * 3]

    // ==================== STREAM B (Deep Strategic Brain) ====================
    /// B1. T4/T5 Shunting Inhibition Optic Encoder:
    /// w_exc: [stream_b_dim * sq_feat_dim], w_inh: [stream_b_dim * sq_feat_dim],
    /// g_leak: [stream_b_dim], w_lc: [stream_b_dim * sq_feat_dim], b_optic: [stream_b_dim].
    pub b_shunt_exc: Vec<f32>,
    pub b_shunt_inh: Vec<f32>,
    pub b_shunt_leak: Vec<f32>,
    pub b_lc_w: Vec<f32>,
    pub b_optic_b: Vec<f32>,

    /// B2. Rich-Club Hub Core (Integrators + Broadcasters):
    pub b_hub_init: Vec<f32>,      // [(num_integrators + num_broadcasters) * stream_b_dim]
    pub b_int_q: Vec<f32>,         // [stream_b_dim * stream_b_dim]
    pub b_int_k: Vec<f32>,         // [stream_b_dim * stream_b_dim]
    pub b_int_v: Vec<f32>,         // [stream_b_dim * stream_b_dim]
    pub b_int_o: Vec<f32>,         // [stream_b_dim * stream_b_dim]
    pub b_rc_mix1: Vec<f32>,       // [stream_b_dim * stream_b_dim]
    pub b_rc_mix2: Vec<f32>,       // [stream_b_dim * stream_b_dim]
    pub b_rc_ln_g: Vec<f32>,       // [stream_b_dim]
    pub b_rc_ln_b: Vec<f32>,       // [stream_b_dim]

    /// B3. Central Complex Phasor Engine (weight-tied across T = 1..6 ticks):
    /// PFN force projection from board summary + integrators into 16 compass sectors x cx_dim.
    pub b_pfn_stm: Vec<f32>,       // [cx_sectors * cx_dim * stream_b_dim]
    pub b_pfn_opp: Vec<f32>,       // [cx_sectors * cx_dim * stream_b_dim]
    /// vDelta vertical composition & hDelta 180-deg phase subtraction: [cx_dim * cx_dim].
    pub b_vdelta: Vec<f32>,
    pub b_hdelta: Vec<f32>,
    /// Delta7 inhibitory ring sharpener: [cx_dim * cx_dim].
    pub b_delta7: Vec<f32>,
    /// CX Ring/Tangential context inhibition fed by [integrator_mean (D) || mbon (32)]:
    /// [cx_dim * (stream_b_dim + mbon_dim)].
    pub b_ring_inh: Vec<f32>,
    pub b_ring_b: Vec<f32>,        // [cx_dim]

    /// B4. Visual Centrifugal Top-Down Inhibitory Gate (71.8% I -> 64 squares) & Broadcaster back to board:
    pub b_cent_w: Vec<f32>,        // [64 * stream_b_dim]
    pub b_cent_b: Vec<f32>,        // [64]
    pub b_broad_to_sq: Vec<f32>,   // [stream_b_dim * stream_b_dim]
    pub b_sq_upd: Vec<f32>,        // [stream_b_dim * stream_b_dim]
    pub b_sq_ln_g: Vec<f32>,       // [stream_b_dim]
    pub b_sq_ln_b: Vec<f32>,       // [stream_b_dim]

    /// B5. K = 4 Creative Orthogonal Goal Plan Bank + PFL3 Policy + PFL2 Gain + Noduli + Pressure + Seed:
    pub b_plan_bank: Vec<f32>,     // [num_creative_plans * cx_dim * (cx_dim + stream_b_dim)]
    pub b_plan_b: Vec<f32>,        // [num_creative_plans * cx_dim]
    /// PFL3 99% contralateral bilinear move policy: Q_from, K_to, K_opp_def: [cx_dim * stream_b_dim].
    pub b_pfl3_q: Vec<f32>,
    pub b_pfl3_k: Vec<f32>,
    pub b_pfl3_opp: Vec<f32>,
    /// PFL2 50/50 bilateral sharpness/urgency gain: [stream_b_dim + cx_dim].
    pub b_pfl2_w: Vec<f32>,
    pub b_pfl2_b: f32,
    /// Deep WDL value head: [3 * (stream_b_dim + cx_dim)].
    pub b_wdl_w: Vec<f32>,
    pub b_wdl_b: [f32; 3],
    /// Strategic Pressure 10-20 ply forecast head: [5 * (stream_b_dim + cx_dim)].
    pub b_press_w: Vec<f32>,
    pub b_press_b: [f32; 5],
    /// Noduli Efference-Copy child eval delta predictor: [cx_dim] dot move_embed.
    pub b_noduli_w: Vec<f32>,      // [cx_dim]
    pub b_noduli_b: f32,
    /// Stream B -> Stream A Top-Down Seed & Centrifugal KC Mask low-rank projections:
    /// seed_proj_down: [seed_rank * (stream_b_dim + cx_dim)]
    /// seed_to_gamma:  [gamma_dim * seed_rank]
    /// seed_to_cent:   [acc_dim * seed_rank]
    pub b_seed_down: Vec<f32>,
    pub b_seed_to_gamma: Vec<f32>,
    pub b_seed_to_cent: Vec<f32>,

    /// B6. 3-Layer Untied FlyWire Optic Lobe Stack (Medulla -> Lobula -> Lobula Plate):
    /// Each layer l in 0..3 executes Pre-LN -> Dm/Pm 4-Head 64x64 Retinotopic Attention ->
    /// Pre-LN -> Tm/TmY Columnar SwiGLU FFN (D -> 2D -> D) for 3000+ Elo spatial chess vision!
    pub b_opt_ln1_g: Vec<f32>,     // [OPTIC_LAYERS * stream_b_dim]
    pub b_opt_ln1_b: Vec<f32>,     // [OPTIC_LAYERS * stream_b_dim]
    pub b_opt_q: Vec<f32>,         // [OPTIC_LAYERS * stream_b_dim * stream_b_dim]
    pub b_opt_k: Vec<f32>,         // [OPTIC_LAYERS * stream_b_dim * stream_b_dim]
    pub b_opt_v: Vec<f32>,         // [OPTIC_LAYERS * stream_b_dim * stream_b_dim]
    pub b_opt_o: Vec<f32>,         // [OPTIC_LAYERS * stream_b_dim * stream_b_dim]
    pub b_opt_rel: Vec<f32>,       // [OPTIC_LAYERS * 4 * 64 * 64]
    pub b_opt_ln2_g: Vec<f32>,     // [OPTIC_LAYERS * stream_b_dim]
    pub b_opt_ln2_b: Vec<f32>,     // [OPTIC_LAYERS * stream_b_dim]
    pub b_opt_fg: Vec<f32>,        // [OPTIC_LAYERS * (2 * stream_b_dim) * stream_b_dim]
    pub b_opt_fu: Vec<f32>,        // [OPTIC_LAYERS * (2 * stream_b_dim) * stream_b_dim]
    pub b_opt_fd: Vec<f32>,        // [OPTIC_LAYERS * stream_b_dim * (2 * stream_b_dim)]

    // ==================== SACCADION-5 DEEP FLY-BRAIN GENIUS CIRCUITS ====================
    /// 3-Lobe Closed Cognitive Loop projection: maps rank-8 fast memory (gamma + DPM alpha'/beta')
    /// directly into the 32-dim MBON vector feeding BOTH SMP and Stream B Central Complex!
    pub a_plastic_to_mbon: Vec<f32>, // [mbon_dim * PLASTIC_RANK]
    /// Amacrine (Dm/Pm) Dynamic Position-Conditioned Attention Gate (biological Smolgen):
    /// Low-rank (4 heads * AMACRINE_RANK = 32) dynamic square-to-square bias per Optic layer.
    pub b_ama_u: Vec<f32>,           // [OPTIC_LAYERS * (4 * AMACRINE_RANK) * stream_b_dim]
    pub b_ama_v: Vec<f32>,           // [OPTIC_LAYERS * (4 * AMACRINE_RANK) * stream_b_dim]
    pub b_ama_g: Vec<f32>,           // [OPTIC_LAYERS * (4 * AMACRINE_RANK) * stream_b_dim]
    /// Central Complex Phasor LayerNorm stabilizing deep multi-tick reasoning (T = 4..8+ ticks).
    pub b_cx_ln_g: Vec<f32>,         // [cx_dim]
    pub b_cx_ln_b: Vec<f32>,         // [cx_dim]
    /// FC2 x EPG -> PFL3 Conjunctive Allocentric-to-Egocentric Goal Steering (Nature Feb 2024):
    /// [cx_dim * (2 * cx_dim)] comparing committed FC2 goal with current EPG board heading.
    pub b_fc2_conj: Vec<f32>,

    // ==================== SACCADION-6 FLYWIRE GRAPH-BRAIN + LIVE WORLD MODEL ====================
    /// FlyWire Dale's-Law Typed Directed Chess-Graph GNN (4 edge types: STM Atk/Def +E, Opp Atk/Def -I):
    pub b_gnn_ln_g: Vec<f32>,        // [stream_b_dim]
    pub b_gnn_ln_b: Vec<f32>,        // [stream_b_dim]
    pub b_gnn_w: Vec<f32>,           // [GRAPH_EDGE_TYPES * stream_b_dim * stream_b_dim]
    /// Live Test-Time hDelta Action-Conditioned Latent World Model (promoted from training aux!):
    /// Predicts future latent z_{t+2} from [global_vec (D+CX) || action_embed (CX)] in < 20 us!
    pub b_wm_w1: Vec<f32>,           // [(stream_b_dim + cx_dim) * (stream_b_dim + 2 * cx_dim)]
    pub b_wm_b1: Vec<f32>,           // [stream_b_dim + cx_dim]
    pub b_wm_w2: Vec<f32>,           // [(stream_b_dim + cx_dim) * (stream_b_dim + cx_dim)]

    // ==================== SACCADION-7 DENDRITIC COINCIDENCE GIANT FIBER (GF/DNp01) ====================
    /// Dedicated end-to-end trained Giant Fiber readout directly from frozen alpha/beta HalfKA lobe
    /// using Pairwise Dendritic Coincidence (`clamp(x_lo) * clamp(x_hi)`) -> 8 phase stacks (< 250ns!):
    pub a_gf_w: Vec<f32>,            // [num_phase_stacks * ab_dim]
    pub a_gf_b: Vec<f32>,            // [num_phase_stacks]

    // ==================== SACCADION-8 RETROGRADE FISCHER-GRAPH + 2-PLY MINIMAX WM ====================
    /// FC2 Retrograde Target-Square Goal Saliency predictor: predicts which squares v are future goals!
    pub b_retro_goal_w: Vec<f32>,    // [stream_b_dim]
    pub b_retro_goal_b: f32,
    /// Backward Reverse-Edge (A^T) 6-Channel X-Ray Goal-Backpropagation GNN:
    pub b_retro_ln_g: Vec<f32>,      // [stream_b_dim]
    pub b_retro_ln_b: Vec<f32>,      // [stream_b_dim]
    pub b_retro_w: Vec<f32>,         // [GRAPH_EDGE_TYPES * stream_b_dim * stream_b_dim]
    /// 2-Ply Counterfactual Latent Minimax Opponent Refutation predictor:
    /// Predicts opponent's sharpest counter-action embedding from post-move latent z_1!
    pub b_wm_opp_w: Vec<f32>,        // [cx_dim * (stream_b_dim + cx_dim)]
    pub b_wm_opp_b: Vec<f32>,        // [cx_dim]

    // ==================== SACCADION-9 2-LAYER NNUE GIANT FIBER MLP (v9!) ====================
    /// L1-resident nonlinear Giant Fiber MLP per phase stack: `ab_dim -> 32 -> 32 -> 1` (SCReLU)
    /// added residually to the linear pairwise skip connection (`a_gf_w/b`)!
    pub a_gf_l1_w: Vec<f32>,         // [num_phase_stacks * GF_H1 * ab_dim]
    pub a_gf_l1_b: Vec<f32>,         // [num_phase_stacks * GF_H1]
    pub a_gf_l2_w: Vec<f32>,         // [num_phase_stacks * GF_H2 * GF_H1]
    pub a_gf_l2_b: Vec<f32>,         // [num_phase_stacks * GF_H2]
    pub a_gf_out_w: Vec<f32>,        // [num_phase_stacks * GF_H2]
    pub a_gf_out_b: Vec<f32>,        // [num_phase_stacks]

    // ==================== DERIVED RUNTIME AVX2 INT16/INT8 CACHE (NOT ON DISK) ====================
    /// Pre-quantized INT16 feature transformer & tightly packed alpha/beta Giant Fiber rows.
    pub a_ft_weight_i16: Vec<i16>,
    pub a_ft_ab_i16: Vec<i16>,
    pub a_ft_bias_i16: Vec<i16>,
    pub a_gf_l1_i8: Vec<QuantizedMatrixI8>,
    pub a_gf_mlp_active: bool,
    pub a_mbon_ipsi_i8: Vec<i8>,
    pub a_mbon_ipsi_scale: Vec<f32>,
    pub a_mbon_contra_i8: Vec<i8>,
    pub a_mbon_contra_scale: Vec<f32>,
    /// Pre-quantized per-row INT8 cache for blazing L2-resident Stream B AVX2 matvecs.
    pub sb_i8: StreamBQuantCache,
}

fn rand_uniform_scaled(rng: &mut u64, scale: f32) -> f32 {
    let u = (splitmix64(rng) >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
    ((u as f32) * 2.0 - 1.0) * scale
}

fn alloc_init(len: usize, scale: f32, rng: &mut u64) -> Vec<f32> {
    (0..len).map(|_| rand_uniform_scaled(rng, scale)).collect()
}

/// Deterministic Drosophila SEZ (Subesophageal Zone) innate survival valence prior (in pawn units).
/// Identical to `training/saccadion_tpu/model.py::init_sez_innate_weights`.
pub fn init_sez_innate_weights(sparse_input_dim: usize, num_phase_stacks: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; sparse_input_dim * num_phase_stacks];
    let piece_pawns = [1.00f32, 3.20, 3.30, 5.00, 9.00, 0.00];
    let halfka_dim = 16 * 768;
    for bucket in 0..16usize {
        for rel_c in 0..2usize {
            let sign = if rel_c == 0 { 1.0f32 } else { -1.0f32 };
            for piece in 0..6usize {
                let base_v = piece_pawns[piece];
                for rel_sq in 0..64usize {
                    let f = bucket * 768 + (rel_c * 6 + piece) * 64 + rel_sq;
                    if f >= halfka_dim || f >= sparse_input_dim {
                        continue;
                    }
                    let file = (rel_sq & 7) as f32;
                    let rank = (rel_sq >> 3) as f32;
                    let center = 1.0 - ((file - 3.5).abs() + (rank - 3.5).abs()) / 7.0;
                    let psq_bonus = match piece {
                        0 => 0.06 * rank + 0.04 * center, // Pawn advancement & center
                        1 | 2 => 0.14 * center,           // Minor piece centralization
                        3 => 0.04 * center,               // Rook activity
                        4 => 0.05 * center,               // Queen centralization
                        _ => 0.0,
                    };
                    let val = sign * (base_v + psq_bonus);
                    for s in 0..num_phase_stacks {
                        w[f * num_phase_stacks + s] = val;
                    }
                }
            }
        }
    }
    w
}

impl SaccadionWeights {
    /// Deterministic He/Xavier initialization matching our exact biological sign constraints.
    pub fn init_random(cfg: ModelConfig, seed: u64) -> Self {
        let mut rng = seed;
        let flyhash_claws = build_flyhash_claws(&cfg);
        let (ab_dim, gamma_dim, _) = cfg.lobe_dims_per_hemi();
        let plastic_dim = cfg.plastic_dim_per_hemi();
        let d = cfg.stream_b_dim;
        let cx = cfg.cx_dim;
        let mbon = cfg.mbon_dim;
        let stacks = cfg.num_phase_stacks;

        let s_ft = (1.0 / 32.0f32).sqrt() * 0.15;
        let s_d = (1.0 / d as f32).sqrt() * 0.5;
        let s_cx = (1.0 / cx as f32).sqrt() * 0.5;

        let a_ft_weight = alloc_init(cfg.sparse_input_dim * cfg.acc_dim, s_ft, &mut rng);
        let a_ft_bias = vec![0.05; cfg.acc_dim];
        let a_ft_weight_i16 = quantize_to_i16(&a_ft_weight);
        let a_ft_ab_i16 = pack_ab_lobe_i16(&a_ft_weight_i16, cfg.sparse_input_dim, cfg.acc_dim, ab_dim);
        let a_ft_bias_i16 = quantize_to_i16(&a_ft_bias);
        let a_sez_weight = init_sez_innate_weights(cfg.sparse_input_dim, stacks);
        let a_claw_weight = alloc_init(plastic_dim * cfg.flyhash_claws, 0.35, &mut rng);
        let a_apl_thresh = vec![0.18; cfg.acc_dim];

        let a_mbon_ipsi = alloc_init(stacks * cfg.acc_dim * mbon, (1.0 / (cfg.acc_dim as f32 * 0.1)).sqrt() * 0.25, &mut rng);
        let a_mbon_contra = alloc_init(stacks * cfg.acc_dim * mbon, (1.0 / (cfg.acc_dim as f32 * 0.1)).sqrt() * 0.25, &mut rng);
        let (a_mbon_ipsi_i8, a_mbon_ipsi_scale) = quantize_stacks_to_i8(&a_mbon_ipsi, stacks);
        let (a_mbon_contra_i8, a_mbon_contra_scale) = quantize_stacks_to_i8(&a_mbon_contra, stacks);
        let a_mbon_bias = vec![0.0; stacks * mbon];

        let a_lh_w_inh = alloc_init(cfg.lh_hidden_dim * cfg.lh_input_dim, 0.15, &mut rng);
        let a_lh_b_inh = vec![0.5; cfg.lh_hidden_dim];
        let a_lh_w_out = alloc_init(4 * cfg.lh_hidden_dim, 0.2, &mut rng);

        let smp_in = 2 * mbon + 4;
        let a_smp_w = alloc_init(stacks * 3 * smp_in, (1.0 / smp_in as f32).sqrt() * 0.3, &mut rng);
        let a_smp_b = vec![0.0; stacks * 3];

        let b_int_q = alloc_init(d * d, s_d, &mut rng);
        let b_int_k = alloc_init(d * d, s_d, &mut rng);
        let b_int_v = alloc_init(d * d, s_d, &mut rng);
        let b_int_o = alloc_init(d * d, s_d, &mut rng);
        let b_rc_mix1 = alloc_init(d * d, s_d, &mut rng);
        let b_rc_mix2 = alloc_init(d * d, s_d, &mut rng);
        let b_broad_to_sq = alloc_init(d * d, s_d, &mut rng);
        let b_sq_upd = alloc_init(d * d, s_d, &mut rng);
        let b_opt_q = alloc_init(OPTIC_LAYERS * d * d, s_d, &mut rng);
        let b_opt_k = alloc_init(OPTIC_LAYERS * d * d, s_d, &mut rng);
        let b_opt_v = alloc_init(OPTIC_LAYERS * d * d, s_d, &mut rng);
        let b_opt_o = alloc_init(OPTIC_LAYERS * d * d, s_d * 0.5, &mut rng);
        let b_opt_fg = alloc_init(OPTIC_LAYERS * (2 * d) * d, s_d, &mut rng);
        let b_opt_fu = alloc_init(OPTIC_LAYERS * (2 * d) * d, s_d, &mut rng);
        let b_opt_fd = alloc_init(OPTIC_LAYERS * d * (2 * d), s_d * 0.5, &mut rng);
        let b_gnn_w = alloc_init(GRAPH_EDGE_TYPES * d * d, s_d * 0.5, &mut rng);
        let b_wm_w1 = alloc_init((d + cx) * (d + 2 * cx), s_d, &mut rng);
        let b_wm_w2 = alloc_init((d + cx) * (d + cx), s_d * 0.5, &mut rng);
        let b_retro_goal_w = alloc_init(d, s_d, &mut rng);
        let b_retro_goal_b = 0.0f32;
        let b_retro_ln_g = vec![1.0; d];
        let b_retro_ln_b = vec![0.0; d];
        let b_retro_w = alloc_init(GRAPH_EDGE_TYPES * d * d, s_d * 0.5, &mut rng);
        let b_wm_opp_w = alloc_init(cx * (d + cx), s_d * 0.5, &mut rng);
        let b_wm_opp_b = vec![0.0; cx];
        let a_gf_l1_w = alloc_init(stacks * GF_H1 * ab_dim, (2.0 / ab_dim as f32).sqrt() * 0.5, &mut rng);
        let a_gf_l1_b = vec![0.05; stacks * GF_H1];
        let a_gf_l2_w = alloc_init(stacks * GF_H2 * GF_H1, (2.0 / GF_H1 as f32).sqrt() * 0.5, &mut rng);
        let a_gf_l2_b = vec![0.05; stacks * GF_H2];
        let a_gf_out_w = alloc_init(stacks * GF_H2, (1.0 / GF_H2 as f32).sqrt() * 0.15, &mut rng);
        let a_gf_out_b = vec![0.0; stacks];
        let a_gf_mlp_active = true;
        let a_gf_l1_i8: Vec<QuantizedMatrixI8> = (0..stacks)
            .map(|s| {
                QuantizedMatrixI8::from_f32(
                    &a_gf_l1_w[s * GF_H1 * ab_dim..(s + 1) * GF_H1 * ab_dim],
                    GF_H1,
                    ab_dim,
                )
            })
            .collect();
        let sb_i8 = StreamBQuantCache::build(
            d, cx, &b_gnn_w, &b_retro_w, &b_opt_q, &b_opt_k, &b_opt_v, &b_opt_o,
            &b_opt_fg, &b_opt_fu, &b_opt_fd, &b_int_q, &b_int_k, &b_int_v, &b_int_o,
            &b_rc_mix1, &b_rc_mix2, &b_broad_to_sq, &b_sq_upd, &b_wm_w1, &b_wm_w2,
        );

        Self {
            flyhash_claws,
            a_ft_weight,
            a_ft_bias,
            a_sez_weight,
            a_claw_weight,
            a_apl_beta: [0.85, 0.85],
            a_apl_thresh,
            a_mbon_ipsi,
            a_mbon_contra,
            a_mbon_bias,
            a_lh_w_inh,
            a_lh_b_inh,
            a_lh_w_out,
            a_lh_b_out: [-1.2, -1.5, -1.5, -1.2],
            a_smp_w,
            a_smp_b,

            b_shunt_exc: alloc_init(d * cfg.sq_feat_dim, s_d, &mut rng),
            b_shunt_inh: alloc_init(d * cfg.sq_feat_dim, s_d, &mut rng),
            b_shunt_leak: vec![0.6; d],
            b_lc_w: alloc_init(d * cfg.sq_feat_dim, s_d, &mut rng),
            b_optic_b: vec![0.0; d],

            b_hub_init: alloc_init((cfg.num_integrators + cfg.num_broadcasters) * d, s_d, &mut rng),
            b_int_q,
            b_int_k,
            b_int_v,
            b_int_o,
            b_rc_mix1,
            b_rc_mix2,
            b_rc_ln_g: vec![1.0; d],
            b_rc_ln_b: vec![0.0; d],

            b_pfn_stm: alloc_init(cfg.cx_sectors * cx * d, s_d, &mut rng),
            b_pfn_opp: alloc_init(cfg.cx_sectors * cx * d, s_d, &mut rng),
            b_vdelta: alloc_init(cx * cx, s_cx, &mut rng),
            b_hdelta: alloc_init(cx * cx, s_cx, &mut rng),
            b_delta7: alloc_init(cx * cx, s_cx, &mut rng),
            b_ring_inh: alloc_init(cx * (d + mbon), s_d, &mut rng),
            b_ring_b: vec![0.0; cx],

            b_cent_w: alloc_init(64 * d, s_d, &mut rng),
            b_cent_b: vec![0.5; 64],
            b_broad_to_sq,
            b_sq_upd,
            b_sq_ln_g: vec![1.0; d],
            b_sq_ln_b: vec![0.0; d],

            b_plan_bank: alloc_init(cfg.num_creative_plans * cx * (cx + d), s_d, &mut rng),
            b_plan_b: vec![0.0; cfg.num_creative_plans * cx],
            b_pfl3_q: alloc_init(cx * d, s_d, &mut rng),
            b_pfl3_k: alloc_init(cx * d, s_d, &mut rng),
            b_pfl3_opp: alloc_init(cx * d, s_d, &mut rng),
            b_pfl2_w: alloc_init(d + cx, s_d, &mut rng),
            b_pfl2_b: 0.0,
            b_wdl_w: alloc_init(3 * (d + cx), s_d, &mut rng),
            b_wdl_b: [0.0; 3],
            b_press_w: alloc_init(5 * (d + cx), s_d, &mut rng),
            b_press_b: [0.0; 5],
            b_noduli_w: alloc_init(cx, s_cx, &mut rng),
            b_noduli_b: 0.0,
            b_seed_down: alloc_init(cfg.seed_rank * (d + cx), s_d, &mut rng),
            b_seed_to_gamma: alloc_init(gamma_dim * cfg.seed_rank, 0.1, &mut rng),
            b_seed_to_cent: alloc_init(cfg.acc_dim * cfg.seed_rank, 0.1, &mut rng),
            b_opt_ln1_g: vec![1.0; OPTIC_LAYERS * d],
            b_opt_ln1_b: vec![0.0; OPTIC_LAYERS * d],
            b_opt_q,
            b_opt_k,
            b_opt_v,
            b_opt_o,
            b_opt_rel: vec![0.0; OPTIC_LAYERS * 4 * 64 * 64],
            b_opt_ln2_g: vec![1.0; OPTIC_LAYERS * d],
            b_opt_ln2_b: vec![0.0; OPTIC_LAYERS * d],
            b_opt_fg,
            b_opt_fu,
            b_opt_fd,
            a_plastic_to_mbon: alloc_init(mbon * PLASTIC_RANK, 0.15, &mut rng),
            b_ama_u: alloc_init(OPTIC_LAYERS * (4 * AMACRINE_RANK) * d, s_d * 0.5, &mut rng),
            b_ama_v: alloc_init(OPTIC_LAYERS * (4 * AMACRINE_RANK) * d, s_d * 0.5, &mut rng),
            b_ama_g: alloc_init(OPTIC_LAYERS * (4 * AMACRINE_RANK) * d, s_d * 0.5, &mut rng),
            b_cx_ln_g: vec![1.0; cx],
            b_cx_ln_b: vec![0.0; cx],
            b_fc2_conj: alloc_init(cx * (2 * cx), s_cx * 0.5, &mut rng),
            b_gnn_ln_g: vec![1.0; d],
            b_gnn_ln_b: vec![0.0; d],
            b_gnn_w,
            b_wm_w1,
            b_wm_b1: vec![0.0; d + cx],
            b_wm_w2,
            a_gf_w: alloc_init(stacks * ab_dim, (1.0 / ab_dim as f32).sqrt() * 0.1, &mut rng),
            a_gf_b: vec![0.0; stacks],
            b_retro_goal_w,
            b_retro_goal_b,
            b_retro_ln_g,
            b_retro_ln_b,
            b_retro_w,
            b_wm_opp_w,
            b_wm_opp_b,
            a_gf_l1_w,
            a_gf_l1_b,
            a_gf_l2_w,
            a_gf_l2_b,
            a_gf_out_w,
            a_gf_out_b,
            a_ft_weight_i16,
            a_ft_ab_i16,
            a_ft_bias_i16,
            a_gf_l1_i8,
            a_gf_mlp_active,
            a_mbon_ipsi_i8,
            a_mbon_ipsi_scale,
            a_mbon_contra_i8,
            a_mbon_contra_scale,
            sb_i8,
            cfg,
        }
    }

    /// Total static parameter count (sparse_params, dense_params).
    pub fn param_counts(&self) -> (usize, usize) {
        let sparse = self.a_ft_weight.len() + self.a_ft_bias.len() + self.a_sez_weight.len();
        let dense = self.a_claw_weight.len()
            + 2
            + self.a_apl_thresh.len()
            + self.a_mbon_ipsi.len()
            + self.a_mbon_contra.len()
            + self.a_mbon_bias.len()
            + self.a_lh_w_inh.len()
            + self.a_lh_b_inh.len()
            + self.a_lh_w_out.len()
            + 4
            + self.a_smp_w.len()
            + self.a_smp_b.len()
            + self.b_shunt_exc.len()
            + self.b_shunt_inh.len()
            + self.b_shunt_leak.len()
            + self.b_lc_w.len()
            + self.b_optic_b.len()
            + self.b_hub_init.len()
            + self.b_int_q.len()
            + self.b_int_k.len()
            + self.b_int_v.len()
            + self.b_int_o.len()
            + self.b_rc_mix1.len()
            + self.b_rc_mix2.len()
            + self.b_rc_ln_g.len()
            + self.b_rc_ln_b.len()
            + self.b_pfn_stm.len()
            + self.b_pfn_opp.len()
            + self.b_vdelta.len()
            + self.b_hdelta.len()
            + self.b_delta7.len()
            + self.b_ring_inh.len()
            + self.b_ring_b.len()
            + self.b_cent_w.len()
            + self.b_cent_b.len()
            + self.b_broad_to_sq.len()
            + self.b_sq_upd.len()
            + self.b_sq_ln_g.len()
            + self.b_sq_ln_b.len()
            + self.b_plan_bank.len()
            + self.b_plan_b.len()
            + self.b_pfl3_q.len()
            + self.b_pfl3_k.len()
            + self.b_pfl3_opp.len()
            + self.b_pfl2_w.len()
            + 1
            + self.b_wdl_w.len()
            + 3
            + self.b_press_w.len()
            + 5
            + self.b_noduli_w.len()
            + 1
            + self.b_seed_down.len()
            + self.b_seed_to_gamma.len()
            + self.b_seed_to_cent.len()
            + self.b_opt_ln1_g.len()
            + self.b_opt_ln1_b.len()
            + self.b_opt_q.len()
            + self.b_opt_k.len()
            + self.b_opt_v.len()
            + self.b_opt_o.len()
            + self.b_opt_rel.len()
            + self.b_opt_ln2_g.len()
            + self.b_opt_ln2_b.len()
            + self.b_opt_fg.len()
            + self.b_opt_fu.len()
            + self.b_opt_fd.len()
            + self.a_plastic_to_mbon.len()
            + self.b_ama_u.len()
            + self.b_ama_v.len()
            + self.b_ama_g.len()
            + self.b_cx_ln_g.len()
            + self.b_cx_ln_b.len()
            + self.b_fc2_conj.len()
            + self.b_gnn_ln_g.len()
            + self.b_gnn_ln_b.len()
            + self.b_gnn_w.len()
            + self.b_wm_w1.len()
            + self.b_wm_b1.len()
            + self.b_wm_w2.len()
            + self.a_gf_w.len()
            + self.a_gf_b.len()
            + self.b_retro_goal_w.len()
            + 1
            + self.b_retro_ln_g.len()
            + self.b_retro_ln_b.len()
            + self.b_retro_w.len()
            + self.b_wm_opp_w.len()
            + self.b_wm_opp_b.len();
        (sparse, dense)
    }

    /// Saves complete weights into canonical `.saccnet` binary format (little-endian FP32 or quantized).
    pub fn save_saccnet(&self, path: &Path) -> std::io::Result<()> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);

        // 80-byte deterministic header matching Python export_rust.py
        w.write_all(&SACCNET_MAGIC)?;
        w.write_all(&SACCNET_VERSION.to_le_bytes())?;
        w.write_all(&self.cfg.tier_id.to_le_bytes())?;
        for &val in &[
            self.cfg.sparse_input_dim as u32,
            self.cfg.acc_dim as u32,
            self.cfg.flyhash_claws as u32,
            self.cfg.mbon_dim as u32,
            self.cfg.num_phase_stacks as u32,
            self.cfg.lh_input_dim as u32,
            self.cfg.lh_hidden_dim as u32,
            self.cfg.sq_feat_dim as u32,
            self.cfg.stream_b_dim as u32,
            self.cfg.num_integrators as u32,
            self.cfg.num_broadcasters as u32,
            self.cfg.cx_sectors as u32,
            self.cfg.cx_dim as u32,
            self.cfg.cx_max_ticks as u32,
            self.cfg.num_creative_plans as u32,
            self.cfg.seed_rank as u32,
        ] {
            w.write_all(&val.to_le_bytes())?;
        }
        w.write_all(&self.cfg.flyhash_seed.to_le_bytes())?;

        let write_slice = |w: &mut BufWriter<File>, s: &[f32]| -> std::io::Result<()> {
            let bytes = unsafe {
                std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4)
            };
            w.write_all(bytes)
        };

        write_slice(&mut w, &self.a_ft_weight)?;
        write_slice(&mut w, &self.a_ft_bias)?;
        write_slice(&mut w, &self.a_sez_weight)?;
        write_slice(&mut w, &self.a_claw_weight)?;
        write_slice(&mut w, &self.a_apl_beta)?;
        write_slice(&mut w, &self.a_apl_thresh)?;
        write_slice(&mut w, &self.a_mbon_ipsi)?;
        write_slice(&mut w, &self.a_mbon_contra)?;
        write_slice(&mut w, &self.a_mbon_bias)?;
        write_slice(&mut w, &self.a_lh_w_inh)?;
        write_slice(&mut w, &self.a_lh_b_inh)?;
        write_slice(&mut w, &self.a_lh_w_out)?;
        write_slice(&mut w, &self.a_lh_b_out)?;
        write_slice(&mut w, &self.a_smp_w)?;
        write_slice(&mut w, &self.a_smp_b)?;

        write_slice(&mut w, &self.b_shunt_exc)?;
        write_slice(&mut w, &self.b_shunt_inh)?;
        write_slice(&mut w, &self.b_shunt_leak)?;
        write_slice(&mut w, &self.b_lc_w)?;
        write_slice(&mut w, &self.b_optic_b)?;
        write_slice(&mut w, &self.b_hub_init)?;
        write_slice(&mut w, &self.b_int_q)?;
        write_slice(&mut w, &self.b_int_k)?;
        write_slice(&mut w, &self.b_int_v)?;
        write_slice(&mut w, &self.b_int_o)?;
        write_slice(&mut w, &self.b_rc_mix1)?;
        write_slice(&mut w, &self.b_rc_mix2)?;
        write_slice(&mut w, &self.b_rc_ln_g)?;
        write_slice(&mut w, &self.b_rc_ln_b)?;
        write_slice(&mut w, &self.b_pfn_stm)?;
        write_slice(&mut w, &self.b_pfn_opp)?;
        write_slice(&mut w, &self.b_vdelta)?;
        write_slice(&mut w, &self.b_hdelta)?;
        write_slice(&mut w, &self.b_delta7)?;
        write_slice(&mut w, &self.b_ring_inh)?;
        write_slice(&mut w, &self.b_ring_b)?;
        write_slice(&mut w, &self.b_cent_w)?;
        write_slice(&mut w, &self.b_cent_b)?;
        write_slice(&mut w, &self.b_broad_to_sq)?;
        write_slice(&mut w, &self.b_sq_upd)?;
        write_slice(&mut w, &self.b_sq_ln_g)?;
        write_slice(&mut w, &self.b_sq_ln_b)?;
        write_slice(&mut w, &self.b_plan_bank)?;
        write_slice(&mut w, &self.b_plan_b)?;
        write_slice(&mut w, &self.b_pfl3_q)?;
        write_slice(&mut w, &self.b_pfl3_k)?;
        write_slice(&mut w, &self.b_pfl3_opp)?;
        write_slice(&mut w, &self.b_pfl2_w)?;
        write_slice(&mut w, &[self.b_pfl2_b])?;
        write_slice(&mut w, &self.b_wdl_w)?;
        write_slice(&mut w, &self.b_wdl_b)?;
        write_slice(&mut w, &self.b_press_w)?;
        write_slice(&mut w, &self.b_press_b)?;
        write_slice(&mut w, &self.b_noduli_w)?;
        write_slice(&mut w, &[self.b_noduli_b])?;
        write_slice(&mut w, &self.b_seed_down)?;
        write_slice(&mut w, &self.b_seed_to_gamma)?;
        write_slice(&mut w, &self.b_seed_to_cent)?;
        write_slice(&mut w, &self.b_opt_ln1_g)?;
        write_slice(&mut w, &self.b_opt_ln1_b)?;
        write_slice(&mut w, &self.b_opt_q)?;
        write_slice(&mut w, &self.b_opt_k)?;
        write_slice(&mut w, &self.b_opt_v)?;
        write_slice(&mut w, &self.b_opt_o)?;
        write_slice(&mut w, &self.b_opt_rel)?;
        write_slice(&mut w, &self.b_opt_ln2_g)?;
        write_slice(&mut w, &self.b_opt_ln2_b)?;
        write_slice(&mut w, &self.b_opt_fg)?;
        write_slice(&mut w, &self.b_opt_fu)?;
        write_slice(&mut w, &self.b_opt_fd)?;
        write_slice(&mut w, &self.a_plastic_to_mbon)?;
        write_slice(&mut w, &self.b_ama_u)?;
        write_slice(&mut w, &self.b_ama_v)?;
        write_slice(&mut w, &self.b_ama_g)?;
        write_slice(&mut w, &self.b_cx_ln_g)?;
        write_slice(&mut w, &self.b_cx_ln_b)?;
        write_slice(&mut w, &self.b_fc2_conj)?;
        write_slice(&mut w, &self.b_gnn_ln_g)?;
        write_slice(&mut w, &self.b_gnn_ln_b)?;
        write_slice(&mut w, &self.b_gnn_w)?;
        write_slice(&mut w, &self.b_wm_w1)?;
        write_slice(&mut w, &self.b_wm_b1)?;
        write_slice(&mut w, &self.b_wm_w2)?;
        write_slice(&mut w, &self.a_gf_w)?;
        write_slice(&mut w, &self.a_gf_b)?;
        write_slice(&mut w, &self.b_retro_goal_w)?;
        write_slice(&mut w, &[self.b_retro_goal_b])?;
        write_slice(&mut w, &self.b_retro_ln_g)?;
        write_slice(&mut w, &self.b_retro_ln_b)?;
        write_slice(&mut w, &self.b_retro_w)?;
        write_slice(&mut w, &self.b_wm_opp_w)?;
        write_slice(&mut w, &self.b_wm_opp_b)?;
        write_slice(&mut w, &self.a_gf_l1_w)?;
        write_slice(&mut w, &self.a_gf_l1_b)?;
        write_slice(&mut w, &self.a_gf_l2_w)?;
        write_slice(&mut w, &self.a_gf_l2_b)?;
        write_slice(&mut w, &self.a_gf_out_w)?;
        write_slice(&mut w, &self.a_gf_out_b)?;

        w.flush()
    }

    /// Loads a `.saccnet` binary weight file exported from JAX TPU training or Rust.
    pub fn load_saccnet(path: &Path) -> std::io::Result<Self> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);

        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if magic != SACCNET_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid SACCNET magic header",
            ));
        }

        let read_u32 = |r: &mut BufReader<File>| -> std::io::Result<u32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        let read_u64 = |r: &mut BufReader<File>| -> std::io::Result<u64> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b))
        };

        let ver = read_u32(&mut r)?;
        if ver < 3 || ver > SACCNET_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported SACCNET version: {} (expected 3..={})", ver, SACCNET_VERSION),
            ));
        }
        let tier_id = read_u32(&mut r)?;
        let cfg = ModelConfig {
            tier_id,
            sparse_input_dim: read_u32(&mut r)? as usize,
            acc_dim: read_u32(&mut r)? as usize,
            flyhash_claws: read_u32(&mut r)? as usize,
            mbon_dim: read_u32(&mut r)? as usize,
            num_phase_stacks: read_u32(&mut r)? as usize,
            lh_input_dim: read_u32(&mut r)? as usize,
            lh_hidden_dim: read_u32(&mut r)? as usize,
            sq_feat_dim: read_u32(&mut r)? as usize,
            stream_b_dim: read_u32(&mut r)? as usize,
            num_integrators: read_u32(&mut r)? as usize,
            num_broadcasters: read_u32(&mut r)? as usize,
            cx_sectors: read_u32(&mut r)? as usize,
            cx_dim: read_u32(&mut r)? as usize,
            cx_max_ticks: read_u32(&mut r)? as usize,
            num_creative_plans: read_u32(&mut r)? as usize,
            seed_rank: read_u32(&mut r)? as usize,
            flyhash_seed: read_u64(&mut r)?,
        };

        let read_vec = |r: &mut BufReader<File>, len: usize| -> std::io::Result<Vec<f32>> {
            let mut v = vec![0.0f32; len];
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len * 4)
            };
            r.read_exact(bytes)?;
            Ok(v)
        };

        let flyhash_claws = build_flyhash_claws(&cfg);
        let (ab_dim, gamma_dim, _) = cfg.lobe_dims_per_hemi();
        let plastic_dim = cfg.plastic_dim_per_hemi();
        let d = cfg.stream_b_dim;
        let cx = cfg.cx_dim;
        let mbon = cfg.mbon_dim;
        let stacks = cfg.num_phase_stacks;

        let a_ft_weight = read_vec(&mut r, cfg.sparse_input_dim * cfg.acc_dim)?;
        let a_ft_bias = read_vec(&mut r, cfg.acc_dim)?;
        let a_sez_weight = if ver >= 4 {
            read_vec(&mut r, cfg.sparse_input_dim * stacks)?
        } else {
            init_sez_innate_weights(cfg.sparse_input_dim, stacks)
        };
        let a_claw_weight = read_vec(&mut r, plastic_dim * cfg.flyhash_claws)?;
        let apl_v = read_vec(&mut r, 2)?;
        let a_apl_thresh = read_vec(&mut r, cfg.acc_dim)?;
        let a_mbon_ipsi = read_vec(&mut r, stacks * cfg.acc_dim * mbon)?;
        let a_mbon_contra = read_vec(&mut r, stacks * cfg.acc_dim * mbon)?;
        let a_mbon_bias = read_vec(&mut r, stacks * mbon)?;
        let a_lh_w_inh = read_vec(&mut r, cfg.lh_hidden_dim * cfg.lh_input_dim)?;
        let a_lh_b_inh = read_vec(&mut r, cfg.lh_hidden_dim)?;
        let a_lh_w_out = read_vec(&mut r, 4 * cfg.lh_hidden_dim)?;
        let lh_bo = read_vec(&mut r, 4)?;
        let a_smp_w = read_vec(&mut r, stacks * 3 * (2 * mbon + 4))?;
        let a_smp_b = read_vec(&mut r, stacks * 3)?;

        let b_shunt_exc = read_vec(&mut r, d * cfg.sq_feat_dim)?;
        let b_shunt_inh = read_vec(&mut r, d * cfg.sq_feat_dim)?;
        let b_shunt_leak = read_vec(&mut r, d)?;
        let b_lc_w = read_vec(&mut r, d * cfg.sq_feat_dim)?;
        let b_optic_b = read_vec(&mut r, d)?;
        let b_hub_init = read_vec(&mut r, (cfg.num_integrators + cfg.num_broadcasters) * d)?;
        let b_int_q = read_vec(&mut r, d * d)?;
        let b_int_k = read_vec(&mut r, d * d)?;
        let b_int_v = read_vec(&mut r, d * d)?;
        let b_int_o = read_vec(&mut r, d * d)?;
        let b_rc_mix1 = read_vec(&mut r, d * d)?;
        let b_rc_mix2 = read_vec(&mut r, d * d)?;
        let b_rc_ln_g = read_vec(&mut r, d)?;
        let b_rc_ln_b = read_vec(&mut r, d)?;
        let b_pfn_stm = read_vec(&mut r, cfg.cx_sectors * cx * d)?;
        let b_pfn_opp = read_vec(&mut r, cfg.cx_sectors * cx * d)?;
        let b_vdelta = read_vec(&mut r, cx * cx)?;
        let b_hdelta = read_vec(&mut r, cx * cx)?;
        let b_delta7 = read_vec(&mut r, cx * cx)?;
        let b_ring_inh = read_vec(&mut r, cx * (d + mbon))?;
        let b_ring_b = read_vec(&mut r, cx)?;
        let b_cent_w = read_vec(&mut r, 64 * d)?;
        let b_cent_b = read_vec(&mut r, 64)?;
        let b_broad_to_sq = read_vec(&mut r, d * d)?;
        let b_sq_upd = read_vec(&mut r, d * d)?;
        let b_sq_ln_g = read_vec(&mut r, d)?;
        let b_sq_ln_b = read_vec(&mut r, d)?;
        let b_plan_bank = read_vec(&mut r, cfg.num_creative_plans * cx * (cx + d))?;
        let b_plan_b = read_vec(&mut r, cfg.num_creative_plans * cx)?;
        let b_pfl3_q = read_vec(&mut r, cx * d)?;
        let b_pfl3_k = read_vec(&mut r, cx * d)?;
        let b_pfl3_opp = read_vec(&mut r, cx * d)?;
        let b_pfl2_w = read_vec(&mut r, d + cx)?;
        let b_pfl2_b = read_vec(&mut r, 1)?[0];
        let b_wdl_w = read_vec(&mut r, 3 * (d + cx))?;
        let wdl_b = read_vec(&mut r, 3)?;
        let b_press_w = read_vec(&mut r, 5 * (d + cx))?;
        let press_b = read_vec(&mut r, 5)?;
        let b_noduli_w = read_vec(&mut r, cx)?;
        let b_noduli_b = read_vec(&mut r, 1)?[0];
        let b_seed_down = read_vec(&mut r, cfg.seed_rank * (d + cx))?;
        let b_seed_to_gamma = read_vec(&mut r, gamma_dim * cfg.seed_rank)?;
        let b_seed_to_cent = read_vec(&mut r, cfg.acc_dim * cfg.seed_rank)?;
        let b_opt_ln1_g = read_vec(&mut r, OPTIC_LAYERS * d)?;
        let b_opt_ln1_b = read_vec(&mut r, OPTIC_LAYERS * d)?;
        let b_opt_q = read_vec(&mut r, OPTIC_LAYERS * d * d)?;
        let b_opt_k = read_vec(&mut r, OPTIC_LAYERS * d * d)?;
        let b_opt_v = read_vec(&mut r, OPTIC_LAYERS * d * d)?;
        let b_opt_o = read_vec(&mut r, OPTIC_LAYERS * d * d)?;
        let b_opt_rel = read_vec(&mut r, OPTIC_LAYERS * 4 * 64 * 64)?;
        let b_opt_ln2_g = read_vec(&mut r, OPTIC_LAYERS * d)?;
        let b_opt_ln2_b = read_vec(&mut r, OPTIC_LAYERS * d)?;
        let b_opt_fg = read_vec(&mut r, OPTIC_LAYERS * (2 * d) * d)?;
        let b_opt_fu = read_vec(&mut r, OPTIC_LAYERS * (2 * d) * d)?;
        let b_opt_fd = read_vec(&mut r, OPTIC_LAYERS * d * (2 * d))?;

        let ama_len = OPTIC_LAYERS * (4 * AMACRINE_RANK) * d;
        let (a_plastic_to_mbon, b_ama_u, b_ama_v, b_ama_g, b_cx_ln_g, b_cx_ln_b, b_fc2_conj) =
            if ver >= 5 {
                (
                    read_vec(&mut r, mbon * PLASTIC_RANK)?,
                    read_vec(&mut r, ama_len)?,
                    read_vec(&mut r, ama_len)?,
                    read_vec(&mut r, ama_len)?,
                    read_vec(&mut r, cx)?,
                    read_vec(&mut r, cx)?,
                    read_vec(&mut r, cx * (2 * cx))?,
                )
            } else {
                // Backward-compatible neutral defaults when loading v3/v4 .saccnet weights
                (
                    vec![0.0; mbon * PLASTIC_RANK],
                    vec![0.0; ama_len],
                    vec![0.0; ama_len],
                    vec![0.0; ama_len],
                    vec![1.0; cx],
                    vec![0.0; cx],
                    vec![0.0; cx * (2 * cx)],
                )
            };

        let (b_gnn_ln_g, b_gnn_ln_b, b_gnn_w, b_wm_w1, b_wm_b1, b_wm_w2) = if ver >= 6 {
            let g_ln_g = read_vec(&mut r, d)?;
            let g_ln_b = read_vec(&mut r, d)?;
            let g_w = if ver >= 8 {
                read_vec(&mut r, GRAPH_EDGE_TYPES * d * d)?
            } else {
                let mut w4 = read_vec(&mut r, 4 * d * d)?;
                w4.resize(GRAPH_EDGE_TYPES * d * d, 0.0);
                w4
            };
            (
                g_ln_g,
                g_ln_b,
                g_w,
                read_vec(&mut r, (d + cx) * (d + 2 * cx))?,
                read_vec(&mut r, d + cx)?,
                read_vec(&mut r, (d + cx) * (d + cx))?,
            )
        } else {
            (
                vec![1.0; d],
                vec![0.0; d],
                vec![0.0; GRAPH_EDGE_TYPES * d * d],
                vec![0.0; (d + cx) * (d + 2 * cx)],
                vec![0.0; d + cx],
                vec![0.0; (d + cx) * (d + cx)],
            )
        };

        let (a_gf_w, a_gf_b) = if ver >= 7 {
            (read_vec(&mut r, stacks * ab_dim)?, read_vec(&mut r, stacks)?)
        } else {
            (vec![0.0; stacks * ab_dim], vec![0.0; stacks])
        };

        let (
            b_retro_goal_w,
            b_retro_goal_b,
            b_retro_ln_g,
            b_retro_ln_b,
            b_retro_w,
            b_wm_opp_w,
            b_wm_opp_b,
        ) = if ver >= 8 {
            (
                read_vec(&mut r, d)?,
                read_vec(&mut r, 1)?[0],
                read_vec(&mut r, d)?,
                read_vec(&mut r, d)?,
                read_vec(&mut r, GRAPH_EDGE_TYPES * d * d)?,
                read_vec(&mut r, cx * (d + cx))?,
                read_vec(&mut r, cx)?,
            )
        } else {
            (
                vec![0.0; d],
                0.0,
                vec![1.0; d],
                vec![0.0; d],
                vec![0.0; GRAPH_EDGE_TYPES * d * d],
                vec![0.0; cx * (d + cx)],
                vec![0.0; cx],
            )
        };

        let (a_gf_l1_w, a_gf_l1_b, a_gf_l2_w, a_gf_l2_b, a_gf_out_w, a_gf_out_b, a_gf_mlp_active) =
            if ver >= 9 {
                (
                    read_vec(&mut r, stacks * GF_H1 * ab_dim)?,
                    read_vec(&mut r, stacks * GF_H1)?,
                    read_vec(&mut r, stacks * GF_H2 * GF_H1)?,
                    read_vec(&mut r, stacks * GF_H2)?,
                    read_vec(&mut r, stacks * GF_H2)?,
                    read_vec(&mut r, stacks)?,
                    true,
                )
            } else {
                (
                    vec![0.0; stacks * GF_H1 * ab_dim],
                    vec![0.0; stacks * GF_H1],
                    vec![0.0; stacks * GF_H2 * GF_H1],
                    vec![0.0; stacks * GF_H2],
                    vec![0.0; stacks * GF_H2],
                    vec![0.0; stacks],
                    false,
                )
            };

        let a_ft_weight_i16 = quantize_to_i16(&a_ft_weight);
        let a_ft_ab_i16 = pack_ab_lobe_i16(&a_ft_weight_i16, cfg.sparse_input_dim, cfg.acc_dim, ab_dim);
        let a_ft_bias_i16 = quantize_to_i16(&a_ft_bias);
        let a_gf_l1_i8: Vec<QuantizedMatrixI8> = (0..stacks)
            .map(|s| {
                QuantizedMatrixI8::from_f32(
                    &a_gf_l1_w[s * GF_H1 * ab_dim..(s + 1) * GF_H1 * ab_dim],
                    GF_H1,
                    ab_dim,
                )
            })
            .collect();
        let (a_mbon_ipsi_i8, a_mbon_ipsi_scale) = quantize_stacks_to_i8(&a_mbon_ipsi, stacks);
        let (a_mbon_contra_i8, a_mbon_contra_scale) = quantize_stacks_to_i8(&a_mbon_contra, stacks);
        let sb_i8 = StreamBQuantCache::build(
            d, cx, &b_gnn_w, &b_retro_w, &b_opt_q, &b_opt_k, &b_opt_v, &b_opt_o,
            &b_opt_fg, &b_opt_fu, &b_opt_fd, &b_int_q, &b_int_k, &b_int_v, &b_int_o,
            &b_rc_mix1, &b_rc_mix2, &b_broad_to_sq, &b_sq_upd, &b_wm_w1, &b_wm_w2,
        );

        Ok(Self {
            cfg,
            flyhash_claws,
            a_ft_weight,
            a_ft_bias,
            a_sez_weight,
            a_claw_weight,
            a_apl_beta: [apl_v[0], apl_v[1]],
            a_apl_thresh,
            a_mbon_ipsi,
            a_mbon_contra,
            a_mbon_bias,
            a_lh_w_inh,
            a_lh_b_inh,
            a_lh_w_out,
            a_lh_b_out: [lh_bo[0], lh_bo[1], lh_bo[2], lh_bo[3]],
            a_smp_w,
            a_smp_b,
            b_shunt_exc,
            b_shunt_inh,
            b_shunt_leak,
            b_lc_w,
            b_optic_b,
            b_hub_init,
            b_int_q,
            b_int_k,
            b_int_v,
            b_int_o,
            b_rc_mix1,
            b_rc_mix2,
            b_rc_ln_g,
            b_rc_ln_b,
            b_pfn_stm,
            b_pfn_opp,
            b_vdelta,
            b_hdelta,
            b_delta7,
            b_ring_inh,
            b_ring_b,
            b_cent_w,
            b_cent_b,
            b_broad_to_sq,
            b_sq_upd,
            b_sq_ln_g,
            b_sq_ln_b,
            b_plan_bank,
            b_plan_b,
            b_pfl3_q,
            b_pfl3_k,
            b_pfl3_opp,
            b_pfl2_w,
            b_pfl2_b,
            b_wdl_w,
            b_wdl_b: [wdl_b[0], wdl_b[1], wdl_b[2]],
            b_press_w,
            b_press_b: [press_b[0], press_b[1], press_b[2], press_b[3], press_b[4]],
            b_noduli_w,
            b_noduli_b,
            b_seed_down,
            b_seed_to_gamma,
            b_seed_to_cent,
            b_opt_ln1_g,
            b_opt_ln1_b,
            b_opt_q,
            b_opt_k,
            b_opt_v,
            b_opt_o,
            b_opt_rel,
            b_opt_ln2_g,
            b_opt_ln2_b,
            b_opt_fg,
            b_opt_fu,
            b_opt_fd,
            a_plastic_to_mbon,
            b_ama_u,
            b_ama_v,
            b_ama_g,
            b_cx_ln_g,
            b_cx_ln_b,
            b_fc2_conj,
            b_gnn_ln_g,
            b_gnn_ln_b,
            b_gnn_w,
            b_wm_w1,
            b_wm_b1,
            b_wm_w2,
            a_gf_w,
            a_gf_b,
            b_retro_goal_w,
            b_retro_goal_b,
            b_retro_ln_g,
            b_retro_ln_b,
            b_retro_w,
            b_wm_opp_w,
            b_wm_opp_b,
            a_gf_l1_w,
            a_gf_l1_b,
            a_gf_l2_w,
            a_gf_l2_b,
            a_gf_out_w,
            a_gf_out_b,
            a_ft_weight_i16,
            a_ft_ab_i16,
            a_ft_bias_i16,
            a_gf_l1_i8,
            a_gf_mlp_active,
            a_mbon_ipsi_i8,
            a_mbon_ipsi_scale,
            a_mbon_contra_i8,
            a_mbon_contra_scale,
            sb_i8,
        })
    }
}
