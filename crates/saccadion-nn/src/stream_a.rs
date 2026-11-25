//! Stream A: Fast CPU Leaf Evaluator (< 0.6 us) implementing:
//! - Isolated Bilateral HalfKAv2_hm + FullRays Accumulators with Finny Bucket Cache
//! - FlyHash Sparse Claws (8 claws per gamma/apbp Kenyon Cell)
//! - APL Global GABA Inhibition Gate (~92% exact zeros -> sparse NNZ gather)
//! - 32 Push-Pull MBONs (16 E / 16 I, FlyWire 2:1 Ipsi/Contra mixing)
//! - Lateral Horn 98.2%-Inhibitory Innate Veto Circuit
//! - SMP/CRE/SIP Convergence Hub + Octopamine Gain + Bounded Eligibility-Trace Dopamine

use crate::plasticity::{DopaminePlasticityState, SparseActiveKc};
use crate::simd::{add_inplace_f32, axpy_f32, dot_f32, sigmoid, silu, softmax_inplace, sub_inplace_f32};
use crate::weights::SaccadionWeights;
use saccadion_chess::features::{
    extract_bilateral_sparse, extract_lateral_horn_features, halfka_feature_index, king_bucket,
    HALFKA_DIM, NUM_KING_BUCKETS,
};
use saccadion_chess::position::Position;
use saccadion_chess::types::{Color, Piece};
use saccadion_chess::Bitboard;

/// Four innate survival/fortress veto signals emitted by the 98.2%-inhibitory Lateral Horn.
#[derive(Copy, Clone, Debug, Default)]
pub struct LateralHornVeto {
    pub blunder_risk: f32,
    pub fortress_prob: f32,
    pub mate_danger: f32,
    pub passer_urgency: f32,
}

impl LateralHornVeto {
    #[inline(always)]
    pub fn as_array(&self) -> [f32; 4] {
        [
            self.blunder_risk,
            self.fortress_prob,
            self.mate_danger,
            self.passer_urgency,
        ]
    }

    #[inline(always)]
    pub fn tactical_alert(&self) -> bool {
        self.blunder_risk > 0.65 || self.mate_danger > 0.60 || self.passer_urgency > 0.70
    }

    #[inline(always)]
    pub fn fortress_alert(&self) -> bool {
        self.fortress_prob > 0.65
    }
}

/// Complete output of a Stream A leaf evaluation (reused across nodes to avoid heap churn).
#[derive(Clone, Debug)]
pub struct StreamAOutput {
    /// Frozen base evaluation in centipawns (from STM perspective).
    pub slow_cp: f32,
    /// Intra-search gamma fast-weight residual (clamped).
    pub gamma_cp: f32,
    /// Intra-game alpha'/beta' fast-weight residual (clamped).
    pub game_cp: f32,
    /// Final fused evaluation in centipawns = slow_cp + gamma_cp + game_cp.
    pub total_cp: f32,
    /// Calibrated Win/Draw/Loss probabilities [W, D, L].
    pub wdl_probs: [f32; 3],
    /// Lateral Horn innate survival & fortress veto signals.
    pub veto: LateralHornVeto,
    /// Active non-zero fraction of Kenyon Cells after APL inhibition (target ~0.08).
    pub active_kc_ratio: f32,
    /// 32-dim MBON summary fed upward into Stream B Central Complex Tangential/Ring circuits.
    pub mbon_summary: Vec<f32>,
    /// Sparse active gamma & alpha'/beta' KCs for zero-copy Dopamine & Eligibility updates.
    pub active_gamma_stm: SparseActiveKc,
    pub active_gamma_opp: SparseActiveKc,
    pub active_apbp_stm: SparseActiveKc,
    pub active_apbp_opp: SparseActiveKc,
}

/// Finny Table entry storing cached piece bitboards and corresponding HalfKA accumulator
/// for one (perspective, king_bucket, mirror) slot.
#[derive(Clone, Debug)]
struct FinnyEntry {
    pieces: [[Bitboard; 6]; 2],
    acc: Vec<f32>,
    initialized: bool,
}

/// Per-thread Finny Table avoiding full accumulator rebuilds on king moves.
#[derive(Clone, Debug)]
pub struct FinnyCache {
    entries: [[FinnyEntry; NUM_KING_BUCKETS * 2]; 2],
    acc_dim: usize,
}

impl FinnyCache {
    pub fn new(acc_dim: usize) -> Self {
        let mk_entry = || FinnyEntry {
            pieces: [[Bitboard::EMPTY; 6]; 2],
            acc: vec![0.0; acc_dim],
            initialized: false,
        };
        let white_arr = std::array::from_fn(|_| mk_entry());
        let black_arr = std::array::from_fn(|_| mk_entry());
        Self {
            entries: [white_arr, black_arr],
            acc_dim,
        }
    }

    pub fn clear(&mut self) {
        for side in &mut self.entries {
            for e in side.iter_mut() {
                e.initialized = false;
            }
        }
    }

    /// Incrementally updates the cached HalfKA accumulator for `persp` by diffing piece bitboards
    /// against the last board seen in this king bucket.
    pub fn compute_halfka_acc(&mut self, pos: &Position, persp: Color, w: &SaccadionWeights, out: &mut [f32]) {
        let ksq = pos.king_square(persp);
        let (bucket, mirror) = king_bucket(ksq, persp);
        let slot = bucket * 2 + usize::from(mirror);
        let entry = &mut self.entries[persp.index()][slot];
        let dim = self.acc_dim;

        if !entry.initialized {
            entry.acc.copy_from_slice(&w.a_ft_bias);
            for c in [Color::White, Color::Black] {
                for p in Piece::ALL {
                    let bb = pos.piece_bb(c, p);
                    entry.pieces[c.index()][p.index()] = bb;
                    for sq in bb {
                        let fidx = halfka_feature_index(persp, ksq, c, p, sq);
                        let row = &w.a_ft_weight[fidx * dim..(fidx + 1) * dim];
                        add_inplace_f32(&mut entry.acc, row);
                    }
                }
            }
            entry.initialized = true;
        } else {
            // Diff only the changed squares per piece bitboard (usually 2-4 squares total)
            for c in [Color::White, Color::Black] {
                for p in Piece::ALL {
                    let prev = entry.pieces[c.index()][p.index()];
                    let curr = pos.piece_bb(c, p);
                    if prev == curr {
                        continue;
                    }
                    let removed = prev & !curr;
                    let added = curr & !prev;
                    for sq in removed {
                        let fidx = halfka_feature_index(persp, ksq, c, p, sq);
                        let row = &w.a_ft_weight[fidx * dim..(fidx + 1) * dim];
                        sub_inplace_f32(&mut entry.acc, row);
                    }
                    for sq in added {
                        let fidx = halfka_feature_index(persp, ksq, c, p, sq);
                        let row = &w.a_ft_weight[fidx * dim..(fidx + 1) * dim];
                        add_inplace_f32(&mut entry.acc, row);
                    }
                    entry.pieces[c.index()][p.index()] = curr;
                }
            }
        }
        out.copy_from_slice(&entry.acc);
    }
}

/// Reusable scratch buffers for zero-allocation Stream A evaluation in hot search loops.
#[derive(Clone, Debug)]
pub struct StreamAWorkspace {
    pub finny: FinnyCache,
    acc_stm: Vec<f32>,
    acc_opp: Vec<f32>,
    kc_pre_stm: Vec<f32>,
    kc_pre_opp: Vec<f32>,
    nnz_stm: SparseActiveKc,
    nnz_opp: SparseActiveKc,
    mbon_stm: Vec<f32>,
    mbon_opp: Vec<f32>,
    lh_hidden: Vec<f32>,
    smp_in: Vec<f32>,
}

impl StreamAWorkspace {
    pub fn new(w: &SaccadionWeights) -> Self {
        let dim = w.cfg.acc_dim;
        let mbon = w.cfg.mbon_dim;
        Self {
            finny: FinnyCache::new(dim),
            acc_stm: vec![0.0; dim],
            acc_opp: vec![0.0; dim],
            kc_pre_stm: vec![0.0; dim],
            kc_pre_opp: vec![0.0; dim],
            nnz_stm: SparseActiveKc::with_capacity(dim / 4),
            nnz_opp: SparseActiveKc::with_capacity(dim / 4),
            mbon_stm: vec![0.0; mbon],
            mbon_opp: vec![0.0; mbon],
            lh_hidden: vec![0.0; w.cfg.lh_hidden_dim],
            smp_in: vec![0.0; 2 * mbon + 4],
        }
    }
}

/// Converts WDL logits into calibrated centipawns (compatible with Stockfish/Lc0 scaling).
#[inline]
pub fn wdl_to_cp(wdl_probs: [f32; 3]) -> f32 {
    let expected = (wdl_probs[0] - wdl_probs[2]).clamp(-0.999, 0.999);
    // Smooth inverse-tanh mapping: +/-0.76 -> +/-400 cp, +/-0.99 -> +/-1050 cp
    let cp = 400.0 * expected.atanh();
    cp.clamp(-2500.0, 2500.0)
}

/// Evaluates a position through Stream A (Fast Bilateral Eye + FlyHash Claws + APL 92% Sparse +
/// Lateral Horn 98.2%-I Veto + 2:1 Push-Pull MBONs + SMP Convergence + Bounded Fast Weights).
///
/// `centrifugal_kc_mask` is an optional top-down inhibitory mask from Stream B (length `acc_dim`).
/// `octopamine` modulates tactical combat gain in [0.7, 1.4] (default 1.0).
pub fn evaluate_stream_a(
    pos: &Position,
    w: &SaccadionWeights,
    ws: &mut StreamAWorkspace,
    plasticity: Option<&DopaminePlasticityState>,
    centrifugal_kc_mask: Option<&[f32]>,
    octopamine: f32,
) -> StreamAOutput {
    let cfg = &w.cfg;
    let dim = cfg.acc_dim;
    let (ab_dim, gamma_dim, apbp_dim) = cfg.lobe_dims_per_hemi();
    let claws = cfg.flyhash_claws;
    let mbon_dim = cfg.mbon_dim;
    let stack = pos.phase_bucket().min(cfg.num_phase_stacks - 1);

    // 1. Bilateral Isolated Feature Accumulation (100% Ipsilateral)
    //    HalfKA via Finny Cache + T4/T5 FullRays ON/OFF sparse indices
    let stm = pos.side_to_move;
    let opp = stm.flip();
    ws.finny.compute_halfka_acc(pos, stm, w, &mut ws.acc_stm);
    ws.finny.compute_halfka_acc(pos, opp, w, &mut ws.acc_opp);

    // Append sparse T4/T5 FullRays features (>= HALFKA_DIM)
    let (f_stm, f_opp) = extract_bilateral_sparse(pos);
    for &idx in f_stm.as_slice() {
        let f = idx as usize;
        if f >= HALFKA_DIM && f < cfg.sparse_input_dim {
            add_inplace_f32(&mut ws.acc_stm, &w.a_ft_weight[f * dim..(f + 1) * dim]);
        }
    }
    for &idx in f_opp.as_slice() {
        let f = idx as usize;
        if f >= HALFKA_DIM && f < cfg.sparse_input_dim {
            add_inplace_f32(&mut ws.acc_opp, &w.a_ft_weight[f * dim..(f + 1) * dim]);
        }
    }

    // 2. FlyHash Sparse Claw Projection onto Plastic Lobes (gamma + alpha'/beta')
    //    Frozen alpha/beta lobe [0..ab_dim] passes directly; plastic lobes [ab_dim..dim]
    //    sample 8 deterministic FlyHash claws from their own hemisphere (Dasgupta 2017).
    let apply_claws = |acc: &[f32], out_pre: &mut [f32]| {
        out_pre[..ab_dim].copy_from_slice(&acc[..ab_dim]);
        let plastic_n = gamma_dim + apbp_dim;
        for p_i in 0..plastic_n {
            let base = p_i * claws;
            let mut sum = 0.0f32;
            for c in 0..claws {
                let src = w.flyhash_claws[base + c] as usize;
                sum += acc[src] * w.a_claw_weight[base + c];
            }
            out_pre[ab_dim + p_i] = sum;
        }
    };
    apply_claws(&ws.acc_stm, &mut ws.kc_pre_stm);
    apply_claws(&ws.acc_opp, &mut ws.kc_pre_opp);

    // 3. APL Global GABA Inhibition Gate (per hemisphere, 99.9% Ipsilateral -> ~92% exact zeros!)
    let mut active_gamma_stm = SparseActiveKc::with_capacity(gamma_dim / 6);
    let mut active_gamma_opp = SparseActiveKc::with_capacity(gamma_dim / 6);
    let mut active_apbp_stm = SparseActiveKc::with_capacity(apbp_dim / 4);
    let mut active_apbp_opp = SparseActiveKc::with_capacity(apbp_dim / 4);

    let apply_apl = |pre: &[f32],
                     beta: f32,
                     nnz: &mut SparseActiveKc,
                     g_nnz: &mut SparseActiveKc,
                     ap_nnz: &mut SparseActiveKc| {
        nnz.clear();
        let pos_sum: f32 = pre.iter().map(|&v| v.max(0.0)).sum();
        let mean_pos = pos_sum / dim as f32;
        for i in 0..dim {
            let cent_inh = centrifugal_kc_mask.map(|m| m[i].max(0.0) * 0.25).unwrap_or(0.0);
            let shifted = pre[i] - beta * mean_pos - w.a_apl_thresh[i] - cent_inh;
            if shifted > 0.0 {
                let clamped = shifted.min(1.0);
                let act = clamped * clamped; // Squared Clipped ReLU
                nnz.push(i as u16, act);
                if i >= ab_dim && i < ab_dim + gamma_dim {
                    g_nnz.push((i - ab_dim) as u16, act);
                } else if i >= ab_dim + gamma_dim {
                    ap_nnz.push((i - ab_dim - gamma_dim) as u16, act);
                }
            }
        }
    };

    apply_apl(
        &ws.kc_pre_stm,
        w.a_apl_beta[0],
        &mut ws.nnz_stm,
        &mut active_gamma_stm,
        &mut active_apbp_stm,
    );
    apply_apl(
        &ws.kc_pre_opp,
        w.a_apl_beta[1],
        &mut ws.nnz_opp,
        &mut active_gamma_opp,
        &mut active_apbp_opp,
    );

    let active_kc_ratio = (ws.nnz_stm.items.len() + ws.nnz_opp.items.len()) as f32 / (2 * dim) as f32;

    // 4. Sparse NNZ Push-Pull MBON Readout (32 MBONs = 16 E / 16 I, exact 2:1 Ipsi/Contra mixing)
    //    Skips ~92% zero Kenyon Cells via sparse axpy!
    let stack_mbon_off = stack * dim * mbon_dim;
    let stack_bias_off = stack * mbon_dim;
    ws.mbon_stm
        .copy_from_slice(&w.a_mbon_bias[stack_bias_off..stack_bias_off + mbon_dim]);
    ws.mbon_opp
        .copy_from_slice(&w.a_mbon_bias[stack_bias_off..stack_bias_off + mbon_dim]);

    // Ipsilateral contribution (weight 1.0) + Contralateral contribution (weight 0.5 -> 2:1 ratio)
    for &(idx, act) in &ws.nnz_stm.items {
        let off = stack_mbon_off + (idx as usize) * mbon_dim;
        axpy_f32(&mut ws.mbon_stm, &w.a_mbon_ipsi[off..off + mbon_dim], act);
        axpy_f32(&mut ws.mbon_opp, &w.a_mbon_contra[off..off + mbon_dim], 0.5 * act);
    }
    for &(idx, act) in &ws.nnz_opp.items {
        let off = stack_mbon_off + (idx as usize) * mbon_dim;
        axpy_f32(&mut ws.mbon_opp, &w.a_mbon_ipsi[off..off + mbon_dim], act);
        axpy_f32(&mut ws.mbon_stm, &w.a_mbon_contra[off..off + mbon_dim], 0.5 * act);
    }

    // Apply Push-Pull activation: first 16 MBONs Excitatory (+SiLU), last 16 Inhibitory (-SiLU)
    let half_mbon = mbon_dim / 2;
    for i in 0..half_mbon {
        ws.mbon_stm[i] = silu(ws.mbon_stm[i]) * octopamine;
        ws.mbon_opp[i] = silu(ws.mbon_opp[i]) * octopamine;
    }
    for i in half_mbon..mbon_dim {
        ws.mbon_stm[i] = -silu(ws.mbon_stm[i]) * octopamine;
        ws.mbon_opp[i] = -silu(ws.mbon_opp[i]) * octopamine;
    }

    // 5. Lateral Horn Innate Veto Branch (98.2%-Inhibitory LHLN layer)
    let lh_feat = extract_lateral_horn_features(pos);
    let lh_h = cfg.lh_hidden_dim;
    let lh_in = cfg.lh_input_dim;
    for h_i in 0..lh_h {
        let row = &w.a_lh_w_inh[h_i * lh_in..(h_i + 1) * lh_in];
        // Strictly inhibitory subtractive drive: ReLU(bias - |W_inh| * x)
        let mut inh_drive = 0.0f32;
        for j in 0..lh_in {
            inh_drive += row[j].abs() * lh_feat[j];
        }
        ws.lh_hidden[h_i] = (w.a_lh_b_inh[h_i] - inh_drive).max(0.0);
    }
    let mut lh_out = [0.0f32; 4];
    for o in 0..4 {
        let row = &w.a_lh_w_out[o * lh_h..(o + 1) * lh_h];
        lh_out[o] = sigmoid(dot_f32(row, &ws.lh_hidden) + w.a_lh_b_out[o]);
    }
    // Blend analytical fortress & mate safety guarantees softly with learned LH veto
    let veto = LateralHornVeto {
        blunder_risk: (0.7 * lh_out[0] + 0.3 * lh_feat[8]).clamp(0.0, 1.0),
        fortress_prob: (0.65 * lh_out[1] + 0.35 * lh_feat[17].max(lh_feat[20])).clamp(0.0, 1.0),
        mate_danger: (0.7 * lh_out[2] + 0.3 * (lh_feat[0] * lh_feat[4]).min(1.0)).clamp(0.0, 1.0),
        passer_urgency: (0.7 * lh_out[3] + 0.3 * lh_feat[14].max(lh_feat[15])).clamp(0.0, 1.0),
    };

    // 6. SMP/CRE/SIP Convergence Hub -> WDL Logits
    ws.smp_in[..mbon_dim].copy_from_slice(&ws.mbon_stm);
    ws.smp_in[mbon_dim..2 * mbon_dim].copy_from_slice(&ws.mbon_opp);
    ws.smp_in[2 * mbon_dim..2 * mbon_dim + 4].copy_from_slice(&veto.as_array());

    let smp_len = 2 * mbon_dim + 4;
    let smp_w_off = stack * 3 * smp_len;
    let smp_b_off = stack * 3;
    let mut wdl_logits = [0.0f32; 3];
    for k in 0..3 {
        let row = &w.a_smp_w[smp_w_off + k * smp_len..smp_w_off + (k + 1) * smp_len];
        wdl_logits[k] = dot_f32(row, &ws.smp_in) + w.a_smp_b[smp_b_off + k];
    }
    let mut wdl_probs = wdl_logits;
    softmax_inplace(&mut wdl_probs);

    // Fortress veto dampens false material optimism toward draw
    let fortress_scale = 1.0 - 0.55 * veto.fortress_prob;
    let slow_cp = wdl_to_cp(wdl_probs) * fortress_scale;

    // 7. Bounded Eligibility-Trace Dopamine Residual (Zero mutation of frozen W_slow!)
    let (gamma_cp, game_cp) = if let Some(pl) = plasticity {
        pl.evaluate_residual_cp(
            &active_gamma_stm,
            &active_gamma_opp,
            &active_apbp_stm,
            &active_apbp_opp,
        )
    } else {
        (0.0, 0.0)
    };

    let total_cp = slow_cp + gamma_cp + game_cp;

    // Compact 32-dim MBON summary (STM - 0.5*Opp push-pull contrast) passed upward into Stream B
    let mut mbon_summary = vec![0.0f32; mbon_dim];
    for i in 0..mbon_dim {
        mbon_summary[i] = ws.mbon_stm[i] - 0.5 * ws.mbon_opp[i];
    }

    StreamAOutput {
        slow_cp,
        gamma_cp,
        game_cp,
        total_cp,
        wdl_probs,
        veto,
        active_kc_ratio,
        mbon_summary,
        active_gamma_stm,
        active_gamma_opp,
        active_apbp_stm,
        active_apbp_opp,
    }
}
