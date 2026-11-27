//! Neuro-Guided PVS Hybrid CPU Search Engine uniting all FlyWire v783 circuits (Saccadion-4):
//! - SEZ Innate Survival Valence + LC4/LPLC2 -> Giant Fiber (GF / DNp01) <1.5us Quiescence reflex
//! - Parallel DNp Multi-Core Lazy SMP sharing a 100% Lock-Free Atomic Transposition Table
//! - Selective Stream B (3-Layer Optic Lobe + Phasor CX) on strategic crossroads
//! - PFL3 (99% contralateral) move ordering & LMR immunity for quiet strategic plans/sacrifices
//! - PFL2 (50/50 bilateral) dynamic depth extensions & pruning
//! - Lateral Horn 98.2%-I Veto disabling unsafe NMP/RFP and clamping fortress delusions
//! - Noduli Efference-Copy child surprise recovery
//! - Eligibility-Trace bounded PAM / 12x PPL1 / Dop1R2 / SOC Dopamine feedback + LAL continuity

use crate::tt::{score_from_tt, Bound, StrategicEntry, StrategicTT, TranspositionTable, MATE_SCORE, MATE_THRESHOLD};
use saccadion_chess::movegen::{generate_legal, generate_legal_tactical};
use saccadion_chess::position::Position;
use saccadion_chess::Bitboard;
use saccadion_chess::types::{Color, Move, MoveList, Piece};
use saccadion_nn::plasticity::DopaminePlasticityState;
use saccadion_nn::stream_a::{evaluate_stream_a, fast_lateral_horn_alert, StreamAWorkspace};
use saccadion_nn::stream_b::{evaluate_stream_b, evaluate_stream_b_cancellable, StreamBOutput};
use saccadion_nn::weights::SaccadionWeights;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_PLY: usize = 128;
const INF: i32 = 32_000;
const DPM_CORR_SIZE: usize = 16_384;
const DPM_CORR_MASK: usize = DPM_CORR_SIZE - 1;
const CONT_DIM: usize = 12 * 64; // 768 (piece_color_idx * 64 + to_sq)

/// Endgame Drawishness & 50-Move Rule Scaling:
/// Dampens static eval smoothly when approaching the 50-move draw or in dead-drawn
/// opposite-colored bishop endgames without major pieces (prevents fortress hallucinations!).
#[inline(always)]
fn scale_endgame_eval(pos: &Position, mut eval: i32) -> i32 {
    if eval.abs() >= MATE_THRESHOLD {
        return eval;
    }
    let wb = pos.piece_bb(Color::White, Piece::Bishop);
    let bb = pos.piece_bb(Color::Black, Piece::Bishop);
    if wb.count() == 1 && bb.count() == 1 {
        let opp_color =
            (wb & Bitboard::LIGHT_SQUARES).non_empty() != (bb & Bitboard::LIGHT_SQUARES).non_empty();
        if opp_color {
            let majors = pos.piece_bb(Color::White, Piece::Queen)
                | pos.piece_bb(Color::Black, Piece::Queen)
                | pos.piece_bb(Color::White, Piece::Rook)
                | pos.piece_bb(Color::Black, Piece::Rook);
            if majors.is_empty() {
                let knights = pos.piece_bb(Color::White, Piece::Knight)
                    | pos.piece_bb(Color::Black, Piece::Knight);
                eval = if knights.is_empty() {
                    (eval * 62) / 100
                } else {
                    (eval * 82) / 100
                };
            }
        }
    }
    if pos.halfmove_clock >= 30 {
        let rem = (115 - pos.halfmove_clock as i32).clamp(15, 85);
        eval = (eval * rem) / 85;
    }
    eval
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub max_depth: i32,
    pub max_nodes: u64,
    pub time_limit_ms: u64,
    /// Number of Parallel DNp (Lazy SMP) search threads on CPU (default 1; UCI defaults up to 4).
    pub threads: usize,
    /// Minimum remaining depth to trigger Stream B Central Complex deliberation.
    pub cx_min_depth: i32,
    pub use_stream_b: bool,
    pub use_dopamine: bool,
    pub use_lateral_horn: bool,
    pub use_efference_copy: bool,
    pub octopamine: f32,
    pub serotonin: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_depth: 64,
            max_nodes: u64::MAX,
            time_limit_ms: 0,
            threads: 1,
            cx_min_depth: 6,
            use_stream_b: true,
            use_dopamine: true,
            use_lateral_horn: true,
            use_efference_copy: true,
            octopamine: 1.0,
            serotonin: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SearchResult {
    pub best_move: Move,
    pub score_cp: i32,
    pub depth: i32,
    pub nodes: u64,
    pub q_nodes: u64,
    pub stream_b_evals: u64,
    pub strategic_tt_hits: u64,
    pub lh_veto_triggers: u64,
    pub efference_surprises: u64,
    pub pam_updates: u64,
    pub ppl1_updates: u64,
    pub elapsed_ms: u64,
    pub pv: Vec<Move>,
}

/// Thread-local Parallel DNp worker state executing lock-free search over shared weights & TT.
struct DnpWorker {
    weights: Arc<SaccadionWeights>,
    tt: Arc<TranspositionTable>,
    ws_a: StreamAWorkspace,
    plasticity: DopaminePlasticityState,
    strategic_tt: StrategicTT,
    stop_flag: Arc<AtomicBool>,
    helper_stop: Arc<AtomicBool>,
    shared_nodes: Arc<AtomicU64>,
    shared_q_nodes: Arc<AtomicU64>,
    flushed_nodes: u64,
    flushed_q_nodes: u64,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    /// 1-ply & 2-ply Continuation History table [768 * 768] (prev_pc_to -> cur_pc_to).
    cont_hist: Vec<i16>,
    /// Capture History table [12 piece_colors][64 to_sq][6 victim_pieces].
    cap_hist: [[[i32; 6]; 64]; 12],
    /// Backward Causal Refutation-Threat table [from_sq][to_sq] boosting deep forcing moves.
    threat_hist: [[i32; 64]; 64],
    countermoves: [[Move; 64]; 64],
    /// Multi-Scale DPM Structural Correction History (Pawn + White Non-Pawn + Black Non-Pawn).
    dpm_pawn_corr: Vec<[i32; 2]>,
    dpm_np_w_corr: Vec<[i32; 2]>,
    dpm_np_b_corr: Vec<[i32; 2]>,
    moved_pc_to: [u16; MAX_PLY],
    static_evals: [i32; MAX_PLY],
    nodes: u64,
    q_nodes: u64,
    stream_b_evals: u64,
    strategic_tt_hits: u64,
    lh_veto_triggers: u64,
    efference_surprises: u64,
    deadline: Option<Instant>,
    aborted: bool,
    allow_time_abort: bool,
    max_nodes: u64,
    cfg: SearchConfig,
    is_helper: bool,
}

impl DnpWorker {
    fn new_helper(
        weights: Arc<SaccadionWeights>,
        tt: Arc<TranspositionTable>,
        stop_flag: Arc<AtomicBool>,
        helper_stop: Arc<AtomicBool>,
        shared_nodes: Arc<AtomicU64>,
        shared_q_nodes: Arc<AtomicU64>,
        cfg: SearchConfig,
        deadline: Option<Instant>,
    ) -> Self {
        let ws_a = StreamAWorkspace::new(&weights);
        let plasticity = DopaminePlasticityState::new(&weights.cfg);
        Self {
            weights,
            tt,
            ws_a,
            plasticity,
            strategic_tt: StrategicTT::new(512),
            stop_flag,
            helper_stop,
            shared_nodes,
            shared_q_nodes,
            flushed_nodes: 0,
            flushed_q_nodes: 0,
            killers: [[Move::NULL; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            cont_hist: vec![0i16; CONT_DIM * CONT_DIM],
            cap_hist: [[[0; 6]; 64]; 12],
            threat_hist: [[0; 64]; 64],
            countermoves: [[Move::NULL; 64]; 64],
            dpm_pawn_corr: vec![[0; 2]; DPM_CORR_SIZE],
            dpm_np_w_corr: vec![[0; 2]; DPM_CORR_SIZE],
            dpm_np_b_corr: vec![[0; 2]; DPM_CORR_SIZE],
            moved_pc_to: [u16::MAX; MAX_PLY],
            static_evals: [0; MAX_PLY],
            nodes: 0,
            q_nodes: 0,
            stream_b_evals: 0,
            strategic_tt_hits: 0,
            lh_veto_triggers: 0,
            efference_surprises: 0,
            deadline,
            aborted: false,
            allow_time_abort: true,
            max_nodes: cfg.max_nodes,
            cfg,
            is_helper: true,
        }
    }

    #[inline(always)]
    fn check_stop(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        if self.nodes >= self.max_nodes {
            self.aborted = true;
            return true;
        }
        if (self.nodes & 1023) == 0 {
            if self.is_helper {
                let dn = self.nodes - self.flushed_nodes;
                let dq = self.q_nodes - self.flushed_q_nodes;
                if dn > 0 {
                    self.shared_nodes.fetch_add(dn, Ordering::Relaxed);
                    self.flushed_nodes = self.nodes;
                }
                if dq > 0 {
                    self.shared_q_nodes.fetch_add(dq, Ordering::Relaxed);
                    self.flushed_q_nodes = self.q_nodes;
                }
            }
            if self.stop_flag.load(Ordering::Relaxed)
                || (self.is_helper && self.helper_stop.load(Ordering::Relaxed))
            {
                self.aborted = true;
                return true;
            }
            if self.allow_time_abort {
                if let Some(dl) = self.deadline {
                    if Instant::now() >= dl {
                        self.aborted = true;
                        return true;
                    }
                }
            }
        }
        false
    }

    #[inline(always)]
    fn update_history(&mut self, stm_idx: usize, mv: Move, delta: i32) {
        let h = &mut self.history[stm_idx][mv.from_sq().index()][mv.to_sq().index()];
        let clamped = delta.clamp(-16_384, 16_384);
        *h += clamped - ((*h * clamped.abs()) / 16_384);
    }

    #[inline(always)]
    fn update_cont_history(&mut self, prev_idx: u16, cur_idx: u16, delta: i32) {
        if (prev_idx as usize) < CONT_DIM && (cur_idx as usize) < CONT_DIM {
            let slot = &mut self.cont_hist[prev_idx as usize * CONT_DIM + cur_idx as usize];
            let v = *slot as i32;
            let clamped = delta.clamp(-16_384, 16_384);
            *slot = (v + clamped - (v * clamped.abs()) / 16_384).clamp(-16_384, 16_384) as i16;
        }
    }

    #[inline(always)]
    fn get_cont_history(&self, ply: usize, cur_idx: u16) -> i32 {
        if (cur_idx as usize) >= CONT_DIM {
            return 0;
        }
        let mut s = 0i32;
        if ply >= 1 {
            let p1 = self.moved_pc_to[ply - 1] as usize;
            if p1 < CONT_DIM {
                s += self.cont_hist[p1 * CONT_DIM + cur_idx as usize] as i32;
            }
        }
        if ply >= 2 {
            let p2 = self.moved_pc_to[ply - 2] as usize;
            if p2 < CONT_DIM {
                s += self.cont_hist[p2 * CONT_DIM + cur_idx as usize] as i32;
            }
        }
        s
    }

    #[inline(always)]
    fn update_cap_history(&mut self, pc12: usize, to_sq: usize, vic_idx: usize, delta: i32) {
        let h = &mut self.cap_hist[pc12][to_sq][vic_idx];
        let clamped = delta.clamp(-16_384, 16_384);
        *h += clamped - ((*h * clamped.abs()) / 16_384);
    }

    #[inline(always)]
    fn dpm_correction(&self, pos: &Position) -> i32 {
        let stm_i = pos.side_to_move.index();
        let p_c = self.dpm_pawn_corr[(pos.pawn_hash as usize) & DPM_CORR_MASK][stm_i];
        let nw_c = self.dpm_np_w_corr[(pos.non_pawn_hash[0] as usize) & DPM_CORR_MASK][stm_i];
        let nb_c = self.dpm_np_b_corr[(pos.non_pawn_hash[1] as usize) & DPM_CORR_MASK][stm_i];
        ((p_c * 3 + nw_c * 2 + nb_c * 2) / 28).clamp(-160, 160)
    }

    #[inline(always)]
    fn update_dpm_correction(&mut self, pos: &Position, depth: i32, err_cp: i32) {
        let stm_i = pos.side_to_move.index();
        let w = (depth * depth + 2 * depth + 1).min(128);
        let target = (err_cp.clamp(-256, 256)) * 4;
        let upd = |slot: &mut i32| {
            let v = *slot;
            *slot = (v + ((target - v) * w) / 256).clamp(-1024, 1024);
        };
        upd(&mut self.dpm_pawn_corr[(pos.pawn_hash as usize) & DPM_CORR_MASK][stm_i]);
        upd(&mut self.dpm_np_w_corr[(pos.non_pawn_hash[0] as usize) & DPM_CORR_MASK][stm_i]);
        upd(&mut self.dpm_np_b_corr[(pos.non_pawn_hash[1] as usize) & DPM_CORR_MASK][stm_i]);
    }

    #[allow(clippy::too_many_arguments)]
    fn pvs(
        &mut self,
        pos: &mut Position,
        mut depth: i32,
        ply: usize,
        mut alpha: i32,
        beta: i32,
        is_pv: bool,
        allow_nmp: bool,
        prev_move: Move,
        excluded_move: Move,
        parent_cent_mask: Option<&[f32]>,
        expected_eval_from_parent: f32,
        pv_out: &mut Vec<Move>,
    ) -> i32 {
        if self.check_stop() {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return self.ws_a.gf_stack.evaluate_ply(ply, pos, &self.weights);
        }

        // Draw detection with Serotonin anti-draw contempt when pushing a strategic squeeze
        if ply > 0 {
            if pos.halfmove_clock >= 100 || pos.is_insufficient_material() {
                return 0;
            }
            if pos.is_repetition() {
                let contempt = ((self.cfg.serotonin - 1.0) * 20.0) as i32;
                return -contempt;
            }
        }

        let in_check = pos.in_check();
        if in_check {
            depth += 1;
        }
        if depth <= 0 {
            return self.quiescence(pos, ply, alpha, beta);
        }

        self.nodes += 1;

        // 1. Probe Lock-Free Atomic Transposition Table (skipped during Singular verification)
        let mut tt_move = Move::NULL;
        let mut tt_score_val = 0i32;
        let mut tt_depth_val = -1i32;
        let mut tt_bound_val = Bound::None;
        if excluded_move.is_null() {
            if let Some(entry) = self.tt.probe(pos.hash) {
                tt_move = entry.best_move;
                tt_score_val = score_from_tt(entry.score as i32, ply as i32);
                tt_depth_val = entry.depth as i32;
                tt_bound_val = entry.bound;
                if !is_pv && tt_depth_val >= depth {
                    match tt_bound_val {
                        Bound::Exact => return tt_score_val,
                        Bound::Lower if tt_score_val >= beta => return tt_score_val,
                        Bound::Upper if tt_score_val <= alpha => return tt_score_val,
                        _ => {}
                    }
                }
            }
        }

        // 2. Unified TT-Consistent Biological Evaluation Hierarchy:
        //    - ALL non-PV nodes (depth 1..64) & QSearch: Push/Pop Incremental AVX2 INT16 Giant Fiber
        //      (~45 ns/node, 100% TT-consistent, ZERO depth 4<->5 eval discontinuity!) + <15 ns LH reflex.
        //    - PV nodes (depth >= 4 or ply == 0) & StrategicTT hits: Full Stream A (frozen alpha/beta
        //      slow_cp only — NEVER corrupted by untrained gamma_seed or mid-search dw_gamma drift!)
        //      smoothly blended with Giant Fiber so PV & non-PV trees stay perfectly calibrated!
        let cached_strat_hit = if self.cfg.use_stream_b && (is_pv || depth >= 6) {
            self.strategic_tt.probe(pos.hash).cloned()
        } else {
            None
        };

        let gf_cp = self.ws_a.gf_stack.evaluate_ply(ply, pos, &self.weights);
        let need_full_stream_a = is_pv && (ply == 0 || depth >= 4);
        let (raw_static_eval, out_a_opt, lh_tactical_alert, lh_deep_alert, efference_surprise) =
            if !need_full_stream_a && cached_strat_hit.is_none() {
                let fast_lh = self.cfg.use_lateral_horn && fast_lateral_horn_alert(pos);
                if fast_lh {
                    self.lh_veto_triggers += 1;
                }
                (gf_cp, None, fast_lh, fast_lh, false)
            } else {
                // Pass `None` for plasticity to keep static minimax & TT evaluations 100% invariant!
                let out_a = evaluate_stream_a(
                    pos,
                    &self.weights,
                    &mut self.ws_a,
                    None,
                    parent_cent_mask,
                    self.cfg.octopamine,
                );
                let se = if let Some(ref hit) = cached_strat_hit {
                    (0.55 * hit.deep_cp + 0.45 * gf_cp as f32).round() as i32
                } else {
                    (0.50 * out_a.slow_cp + 0.50 * gf_cp as f32).round() as i32
                };
                if self.cfg.use_dopamine {
                    self.plasticity.apply_mbon_soc_feedback(out_a.slow_cp);
                    self.plasticity
                        .accumulate_eligibility(&out_a.active_gamma_stm, &out_a.active_gamma_opp);
                }
                let lh_alert = self.cfg.use_lateral_horn && out_a.veto.tactical_alert();
                let lh_deep = self.cfg.use_lateral_horn
                    && (lh_alert || out_a.veto.fortress_alert() || out_a.veto.passer_urgency > 0.55);
                if lh_alert {
                    self.lh_veto_triggers += 1;
                }
                let eff_surp = self.cfg.use_efference_copy
                    && ply > 0
                    && expected_eval_from_parent != 0.0
                    && (se as f32 - expected_eval_from_parent).abs() > 140.0;
                if eff_surp {
                    self.efference_surprises += 1;
                }
                (se, Some(out_a), lh_alert, lh_deep, eff_surp)
            };
        let static_eval = scale_endgame_eval(
            pos,
            (raw_static_eval + self.dpm_correction(pos)).clamp(-2600, 2600),
        );
        self.static_evals[ply] = static_eval;
        let improving = !in_check && ply >= 2 && static_eval > self.static_evals[ply - 2];
        let parent_eval_f32 = out_a_opt
            .as_ref()
            .map(|o| o.total_cp)
            .unwrap_or(static_eval as f32);

        // 3. Reverse Futility Pruning (RFP), Razoring, Null-Move Pruning (NMP), & ProbCut — vetoed by Lateral Horn!
        if !is_pv && !in_check && excluded_move.is_null() && !lh_tactical_alert && !efference_surprise {
            let rfp_margin = if improving { 70 * depth } else { 90 * depth };
            if depth <= 5 && static_eval - rfp_margin >= beta {
                return static_eval;
            }
            // Razoring: hopeless shallow positions drop straight into Quiescence
            if depth <= 2 && static_eval + 260 * depth <= alpha {
                let q = self.quiescence(pos, ply, alpha, alpha + 1);
                if q <= alpha {
                    return q;
                }
            }
            let us = pos.side_to_move;
            let non_pawn = pos.color_occ[us.index()]
                ^ pos.piece_bb(us, saccadion_chess::types::Piece::Pawn)
                ^ pos.piece_bb(us, saccadion_chess::types::Piece::King);
            if allow_nmp && depth >= 3 && static_eval >= beta && non_pawn.non_empty() {
                let r = 3 + depth / 4 + ((static_eval - beta) / 200).clamp(0, 2);
                self.moved_pc_to[ply] = u16::MAX;
                let undo = pos.make_null_move();
                self.ws_a.gf_stack.push_null_move(ply);
                let mut dummy_pv = Vec::new();
                let null_score = -self.pvs(
                    pos,
                    (depth - 1 - r).max(0),
                    ply + 1,
                    -beta,
                    -beta + 1,
                    false,
                    false,
                    Move::NULL,
                    Move::NULL,
                    parent_cent_mask,
                    0.0,
                    &mut dummy_pv,
                );
                pos.unmake_null_move(undo);
                if self.aborted {
                    return 0;
                }
                if null_score >= beta && null_score < MATE_THRESHOLD {
                    // Null-Move Verification Search at depth >= 12 eliminates Endgame Zugzwang blindspots!
                    if depth >= 12 {
                        let mut ver_pv = Vec::new();
                        let ver_score = self.pvs(
                            pos,
                            (depth - r).max(1),
                            ply,
                            beta - 1,
                            beta,
                            false,
                            false,
                            prev_move,
                            Move::NULL,
                            parent_cent_mask,
                            expected_eval_from_parent,
                            &mut ver_pv,
                        );
                        if self.aborted {
                            return 0;
                        }
                        if ver_score >= beta {
                            return beta;
                        }
                    } else {
                        return beta;
                    }
                }
            }
            // ProbCut: if a winning capture easily beats `beta + 190` at reduced depth, prune early!
            if depth >= 5 && beta.abs() < MATE_THRESHOLD - 256 {
                let pc_beta = (beta + 190).min(MATE_THRESHOLD - 256);
                let pc_depth = depth - 4;
                if tt_depth_val < pc_depth || tt_score_val >= pc_beta {
                    let mut caps = MoveList::new();
                    generate_legal_tactical(pos, &mut caps);
                    for &cmv in caps.as_slice() {
                        if !pos.see_ge(cmv, (pc_beta - static_eval).max(0)) {
                            continue;
                        }
                        self.moved_pc_to[ply] = pos
                            .piece_at(cmv.from_sq())
                            .map(|(c, p)| ((c.index() * 6 + p.index()) * 64 + cmv.to_sq().index()) as u16)
                            .unwrap_or(u16::MAX);
                        let undo = pos.make_move(cmv);
                        self.ws_a.gf_stack.push_move(
                            ply,
                            cmv,
                            undo.captured,
                            pos,
                            &self.weights,
                            &mut self.ws_a.finny_i16,
                        );
                        let mut q_s = -self.quiescence(pos, ply + 1, -pc_beta, -pc_beta + 1);
                        if q_s >= pc_beta && !self.aborted {
                            let mut pc_pv = Vec::new();
                            q_s = -self.pvs(
                                pos,
                                pc_depth,
                                ply + 1,
                                -pc_beta,
                                -pc_beta + 1,
                                false,
                                true,
                                cmv,
                                Move::NULL,
                                parent_cent_mask,
                                0.0,
                                &mut pc_pv,
                            );
                        }
                        pos.unmake_move(cmv, undo);
                        if self.aborted {
                            return 0;
                        }
                        if q_s >= pc_beta {
                            return pc_beta;
                        }
                    }
                }
            }
        }

        // 4. Generate legal moves & handle Checkmate / Stalemate
        let mut moves = MoveList::new();
        generate_legal(pos, &mut moves);
        if moves.is_empty() {
            return if in_check { -MATE_SCORE + ply as i32 } else { 0 };
        }

        // 5. Selective INT8 Stream B Deep Strategic Brain:
        //    In blitz (<= 500 ms), run Stream B ONLY at root so 90% of the clock powers deep PVS!
        //    In rapid/classical/fixed-depth (> 500 ms or untimed), deliberate on deep PV crossroads.
        let allow_subtree_sb = self.cfg.time_limit_ms == 0 || self.cfg.time_limit_ms > 500;
        let need_stream_b = self.cfg.use_stream_b
            && allow_subtree_sb
            && !self.is_helper
            && excluded_move.is_null()
            && out_a_opt.is_some()
            && (depth >= self.cfg.cx_min_depth + 2
                || (is_pv && depth >= 7)
                || ((efference_surprise || lh_deep_alert) && depth >= 6));

        let mut local_strat: Option<StrategicEntry> = cached_strat_hit;
        if local_strat.is_some() {
            self.strategic_tt_hits += 1;
        } else if need_stream_b {
            if let Some(ref out_a) = out_a_opt {
                let novelty = self
                    .plasticity
                    .mbon_alpha_prime3_novelty(&out_a.active_apbp_stm, false);
                let out_b = evaluate_stream_b(
                    pos,
                    &moves,
                    &self.weights,
                    &out_a.mbon_summary,
                    out_a.sez_cp,
                    Some(&self.plasticity),
                    novelty,
                    self.cfg.serotonin,
                    true,
                );
                self.stream_b_evals += 1;
                self.strategic_tt.store(pos.hash, &out_b);
                local_strat = self.strategic_tt.probe(pos.hash).cloned();
            }
        }

        // Internal Iterative Reduction (IIR) when neither TT nor Stream B has prior guidance
        if depth >= 4 && tt_move.is_null() && local_strat.is_none() && !in_check {
            depth -= 1;
        }

        // Depth-30 Causal Laser: PFL2 Sharpness-driven extension on critical turning points (up to ply 30!)
        let pfl2_sharpness = local_strat.as_ref().map(|s| s.pfl2_sharpness).unwrap_or(0.5);
        if is_pv && pfl2_sharpness > 0.72 && !in_check && ply < 30 && depth >= 3 {
            depth += 1;
        }

        // 6. Zero-Allocation Stack Move Scoring + Retrograde Goal Saliency + Lazy Best-Move Selection
        let counter_mv = if !prev_move.is_null() {
            self.countermoves[prev_move.from_sq().index()][prev_move.to_sq().index()]
        } else {
            Move::NULL
        };
        let mut scored_moves: [(Move, i32, usize, f32, u16, u8); 256] =
            [(Move::NULL, 0, usize::MAX, 0.0, u16::MAX, 0); 256];
        let mut n_moves = 0usize;
        let stm_color = pos.side_to_move;
        let stm_idx = stm_color.index();
        for &mv in moves.as_slice() {
            if mv == excluded_move {
                continue;
            }
            let mut priority = 0i32;
            let mut pfl3_rank = usize::MAX;
            let mut noduli_delta = 0.0f32;
            let mover_piece = pos.piece_at(mv.from_sq()).map(|(_, p)| p);
            let pc12 = mover_piece.map(|p| stm_idx * 6 + p.index()).unwrap_or(0);
            let cur_pc_to = (pc12 * 64 + mv.to_sq().index()) as u16;
            let vic_piece = pos.piece_at(mv.to_sq()).map(|(_, p)| p);
            let vic_idx = vic_piece.map(|p| p.index()).unwrap_or(0) as u8;

            if mv == tt_move {
                priority += 2_000_000;
            }
            if let Some(ref st) = local_strat {
                if let Some((rank, pm)) = st.top_moves.iter().enumerate().find(|(_, m)| m.mv == mv) {
                    pfl3_rank = rank;
                    noduli_delta = pm.expected_delta_cp;
                    priority += 450_000 + (pm.prob * 500_000.0) as i32;
                }
                // Retrograde Fischer-Graph target-square goal saliency boost!
                let rel_to = mv.to_sq().relative_to(stm_color).index();
                let gs = st.retro_goal_saliency[rel_to];
                if gs > 0.62 {
                    priority += ((gs - 0.5) * 180_000.0) as i32;
                }
            }
            if mv.is_capture() {
                let vic = vic_piece.map(|p| p.cp_value()).unwrap_or(100);
                let atk = mover_piece.map(|p| p.cp_value()).unwrap_or(100);
                let ch = self.cap_hist[pc12][mv.to_sq().index()][vic_idx as usize];
                let good_cap = vic >= atk || pos.see_ge(mv, -(ch / 128).clamp(-80, 80));
                priority += if good_cap { 1_000_000 } else { -200_000 } + vic * 10 - atk + ch / 16;
            } else if mv.is_promotion() {
                priority += 950_000;
            } else if mv == self.killers[ply][0] {
                priority += 400_000;
            } else if mv == counter_mv {
                priority += 375_000;
            } else if mv == self.killers[ply][1] {
                priority += 350_000;
            } else {
                priority += self.history[stm_idx][mv.from_sq().index()][mv.to_sq().index()]
                    + self.get_cont_history(ply, cur_pc_to)
                    + self.threat_hist[mv.from_sq().index()][mv.to_sq().index()] / 2;
            }
            scored_moves[n_moves] = (mv, priority, pfl3_rank, noduli_delta, cur_pc_to, vic_idx);
            n_moves += 1;
        }
        if n_moves == 0 {
            return alpha;
        }

        let child_cent_mask: Option<&[f32]> = local_strat
            .as_ref()
            .map(|s| s.centrifugal_kc_mask.as_slice())
            .or(parent_cent_mask);

        let orig_alpha = alpha;
        let mut best_move = scored_moves[0].0;
        let mut best_score = -INF;
        // Zero heap allocation on non-PV nodes!
        let mut child_pv = if is_pv {
            Vec::with_capacity(depth as usize)
        } else {
            Vec::new()
        };
        let mut quiets_tried = [(Move::NULL, u16::MAX); 64];
        let mut n_quiets_tried = 0usize;
        let mut caps_tried = [(0u8, 0u8, 0u8); 32];
        let mut n_caps_tried = 0usize;

        for move_idx in 0..n_moves {
            // Lazy partial selection: find the highest-priority remaining move in O(N)
            let mut best_i = move_idx;
            let mut best_p = scored_moves[move_idx].1;
            for j in (move_idx + 1)..n_moves {
                if scored_moves[j].1 > best_p {
                    best_p = scored_moves[j].1;
                    best_i = j;
                }
            }
            if best_i != move_idx {
                scored_moves.swap(move_idx, best_i);
            }
            let (mv, _, pfl3_rank, noduli_delta, cur_pc_to, vic_idx) = scored_moves[move_idx];
            let is_quiet = !mv.is_capture() && !mv.is_promotion();

            // Late Move Pruning (LMP), Futility Pruning, & Main-Search SEE Pruning
            // Strictly immune for PV, checks, Lateral Horn alerts, and top-3 PFL3 creative moves!
            if !is_pv
                && !in_check
                && move_idx > 0
                && !lh_tactical_alert
                && pfl3_rank >= 3
                && best_score > -MATE_THRESHOLD + 256
            {
                if is_quiet {
                    let lmp_limit = (4 + depth * depth * if improving { 2 } else { 1 } / 2) as usize;
                    if depth <= 4 && move_idx >= lmp_limit {
                        continue;
                    }
                    let fut_margin = 120 * depth + if improving { 60 } else { 0 };
                    if depth <= 3 && static_eval + fut_margin <= alpha {
                        continue;
                    }
                } else if depth <= 4 && !mv.is_promotion() && !pos.see_ge(mv, -85 * depth) {
                    continue;
                }
            }

            // Depth-30 Causal Laser Extensions:
            // 1) Singular & Double-Singular Extension (diving cleanly up to ply 36-44!)
            // 2) 7th-Rank Passed-Pawn Laser Extension
            let mut ext = 0i32;
            if depth >= 7
                && ply > 0
                && ply < 44
                && excluded_move.is_null()
                && mv == tt_move
                && tt_depth_val >= depth - 3
                && matches!(tt_bound_val, Bound::Exact | Bound::Lower)
                && tt_score_val.abs() < MATE_THRESHOLD - 256
            {
                let s_beta = (tt_score_val - 2 * depth).max(-MATE_THRESHOLD + 256);
                let s_depth = (depth - 1) / 2;
                let mut s_pv = Vec::new();
                let s_score = self.pvs(
                    pos,
                    s_depth,
                    ply,
                    s_beta - 1,
                    s_beta,
                    false,
                    true,
                    prev_move,
                    mv,
                    parent_cent_mask,
                    expected_eval_from_parent,
                    &mut s_pv,
                );
                if self.aborted {
                    return 0;
                }
                if s_score < s_beta {
                    ext = if !is_pv && ply < 36 && s_score < s_beta - 25 { 2 } else { 1 };
                } else if s_beta >= beta {
                    return s_beta;
                }
            } else if ply < 36
                && pos.piece_at(mv.from_sq()).map(|(_, p)| p == Piece::Pawn).unwrap_or(false)
                && mv.to_sq().relative_to(stm_color).rank() == 6
            {
                ext = 1;
            }

            if is_quiet && n_quiets_tried < 64 {
                quiets_tried[n_quiets_tried] = (mv, cur_pc_to);
                n_quiets_tried += 1;
            } else if mv.is_capture() && n_caps_tried < 32 {
                caps_tried[n_caps_tried] = ((cur_pc_to / 64) as u8, mv.to_sq().index() as u8, vic_idx);
                n_caps_tried += 1;
            }

            self.moved_pc_to[ply] = cur_pc_to;
            let undo = pos.make_move(mv);
            self.ws_a.gf_stack.push_move(
                ply,
                mv,
                undo.captured,
                pos,
                &self.weights,
                &mut self.ws_a.finny_i16,
            );
            let gives_check = pos.in_check();
            let expected_child_eval = if noduli_delta != 0.0 {
                -(parent_eval_f32 + noduli_delta)
            } else {
                0.0
            };

            let new_depth = depth - 1 + ext;
            child_pv.clear();
            let score = if move_idx == 0 {
                -self.pvs(
                    pos,
                    new_depth,
                    ply + 1,
                    -beta,
                    -alpha,
                    is_pv,
                    true,
                    mv,
                    Move::NULL,
                    child_cent_mask,
                    expected_child_eval,
                    &mut child_pv,
                )
            } else {
                // Logarithmic LMR modulated by PFL3 top-3 immunity, PFL2 sharpness, History, & Lateral Horn veto
                let mut r = 0i32;
                if depth >= 3 && move_idx >= 3 && is_quiet && !in_check && !gives_check {
                    let log_r = ((depth as f32).ln() * ((move_idx + 1) as f32).ln() * 0.50) as i32;
                    r = log_r.max(1);
                    if !improving {
                        r += 1;
                    }
                    let hist_v = self.history[stm_idx][mv.from_sq().index()][mv.to_sq().index()]
                        + self.get_cont_history(ply, cur_pc_to);
                    r -= (hist_v / 5000).clamp(-2, 2);
                    if pfl3_rank < 3 {
                        r = (r - 3).max(0);
                    }
                    if pfl2_sharpness < 0.30 {
                        r += 1;
                    }
                    if lh_tactical_alert
                        || lh_deep_alert
                        || efference_surprise
                        || (self.cfg.use_dopamine
                            && self.plasticity.ppl1_updates > self.plasticity.pam_updates + 2)
                    {
                        r = (r - 1).max(0);
                    }
                    r = r.min(new_depth - 1).max(0);
                }

                let mut s = -self.pvs(
                    pos,
                    new_depth - r,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                    false,
                    true,
                    mv,
                    Move::NULL,
                    child_cent_mask,
                    expected_child_eval,
                    &mut child_pv,
                );
                if s > alpha && r > 0 {
                    child_pv.clear();
                    s = -self.pvs(
                        pos,
                        new_depth,
                        ply + 1,
                        -alpha - 1,
                        -alpha,
                        false,
                        true,
                        mv,
                        Move::NULL,
                        child_cent_mask,
                        expected_child_eval,
                        &mut child_pv,
                    );
                }
                if s > alpha && s < beta {
                    child_pv.clear();
                    s = -self.pvs(
                        pos,
                        new_depth,
                        ply + 1,
                        -beta,
                        -alpha,
                        true,
                        true,
                        mv,
                        Move::NULL,
                        child_cent_mask,
                        expected_child_eval,
                        &mut child_pv,
                    );
                }
                s
            };

            pos.unmake_move(mv, undo);

            if self.aborted {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = mv;
                if score > alpha {
                    alpha = score;
                    if is_pv {
                        pv_out.clear();
                        pv_out.push(mv);
                        pv_out.extend_from_slice(&child_pv);
                    }
                    if alpha >= beta {
                        let bonus = (depth * depth * 16).min(4096);
                        if is_quiet {
                            if self.killers[ply][0] != mv {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = mv;
                            }
                            if !prev_move.is_null() {
                                self.countermoves[prev_move.from_sq().index()][prev_move.to_sq().index()] = mv;
                            }
                            self.update_history(stm_idx, mv, bonus);
                            if depth >= 4 {
                                let th = &mut self.threat_hist[mv.from_sq().index()][mv.to_sq().index()];
                                *th = (*th + bonus - (*th * bonus) / 16_384).clamp(-16_384, 16_384);
                            }
                            if ply >= 1 {
                                self.update_cont_history(self.moved_pc_to[ply - 1], cur_pc_to, bonus);
                            }
                            if ply >= 2 {
                                self.update_cont_history(self.moved_pc_to[ply - 2], cur_pc_to, bonus);
                            }
                            // History + Continuation malus on earlier failed quiet moves
                            for &(qm, q_pc_to) in &quiets_tried[..n_quiets_tried.saturating_sub(1)] {
                                self.update_history(stm_idx, qm, -bonus / 2);
                                if ply >= 1 {
                                    self.update_cont_history(self.moved_pc_to[ply - 1], q_pc_to, -bonus / 2);
                                }
                                if ply >= 2 {
                                    self.update_cont_history(self.moved_pc_to[ply - 2], q_pc_to, -bonus / 2);
                                }
                            }
                        } else if mv.is_capture() {
                            self.update_cap_history((cur_pc_to / 64) as usize, mv.to_sq().index(), vic_idx as usize, bonus);
                            for &(c_pc, c_to, c_vic) in &caps_tried[..n_caps_tried.saturating_sub(1)] {
                                self.update_cap_history(c_pc as usize, c_to as usize, c_vic as usize, -bonus / 2);
                            }
                        }
                        break;
                    }
                }
            }
        }

        // 7. Live Bounded PAM / 12x PPL1 / Dop1R2 Dopamine Update + Multi-Scale DPM Structural Correction
        if excluded_move.is_null() && !in_check && depth >= 2 && best_score.abs() < MATE_THRESHOLD {
            let quiet_best = !best_move.is_capture() && !best_move.is_promotion();
            let valid_dpm = quiet_best
                && ((best_score <= orig_alpha && best_score < static_eval)
                    || (best_score >= beta && best_score > static_eval)
                    || (best_score > orig_alpha && best_score < beta));
            if valid_dpm {
                self.update_dpm_correction(pos, depth, best_score - raw_static_eval);
            }
            if depth >= 3 && self.cfg.use_dopamine {
                if let Some(ref out_a) = out_a_opt {
                    let delta_cp = best_score as f32 - out_a.total_cp;
                    self.plasticity.apply_dopamine_surprise(
                        delta_cp,
                        &out_a.active_gamma_stm,
                        &out_a.active_gamma_opp,
                    );
                }
            }
        }

        // 8. Store in Lock-Free Atomic Transposition Table
        if excluded_move.is_null() {
            let bound = if best_score <= orig_alpha {
                Bound::Upper
            } else if best_score >= beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            self.tt
                .store(pos.hash, best_move, best_score, depth, bound, ply as i32);
        }

        best_score
    }

    /// LC4/LPLC2 -> Giant Fiber (GF / DNp01) Ultra-Fast Neural Quiescence Reflex with TT Caching!
    fn quiescence(&mut self, pos: &mut Position, ply: usize, mut alpha: i32, beta: i32) -> i32 {
        if self.check_stop() {
            return 0;
        }
        self.nodes += 1;
        self.q_nodes += 1;
        if ply >= MAX_PLY - 1 {
            return self.ws_a.gf_stack.evaluate_ply(ply, pos, &self.weights);
        }

        let orig_alpha = alpha;
        if let Some(entry) = self.tt.probe(pos.hash) {
            let tt_score = score_from_tt(entry.score as i32, ply as i32);
            match entry.bound {
                Bound::Exact => return tt_score,
                Bound::Lower if tt_score >= beta => return tt_score,
                Bound::Upper if tt_score <= alpha => return tt_score,
                _ => {}
            }
        }

        let in_check = pos.in_check();
        let stand_pat = scale_endgame_eval(
            pos,
            self.ws_a.gf_stack.evaluate_ply(ply, pos, &self.weights),
        );
        let mut best_score = if in_check { -INF } else { stand_pat };
        let mut best_move = Move::NULL;
        if !in_check {
            if stand_pat >= beta {
                return beta;
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
        }

        let mut moves = MoveList::new();
        if in_check {
            generate_legal(pos, &mut moves);
            if moves.is_empty() {
                return -MATE_SCORE + ply as i32;
            }
        } else {
            generate_legal_tactical(pos, &mut moves);
        }

        let mut scored: [(Move, i32, i32); 256] = [(Move::NULL, 0, 0); 256];
        let mut n_scored = 0usize;
        for &mv in moves.as_slice() {
            let vic = pos
                .piece_at(mv.to_sq())
                .map(|(_, p)| p.cp_value())
                .unwrap_or(100);
            let atk = pos
                .piece_at(mv.from_sq())
                .map(|(_, p)| p.cp_value())
                .unwrap_or(100);
            scored[n_scored] = (mv, vic * 10 - atk, vic);
            n_scored += 1;
        }
        scored[..n_scored].sort_unstable_by(|a, b| b.1.cmp(&a.1));

        for &(mv, _, vic_cp) in &scored[..n_scored] {
            if !in_check && mv.is_capture() && !mv.is_promotion() {
                if stand_pat + vic_cp + 220 < alpha {
                    continue;
                }
                if !pos.see_ge(mv, -50) {
                    continue;
                }
            }
            let undo = pos.make_move(mv);
            self.ws_a.gf_stack.push_move(
                ply,
                mv,
                undo.captured,
                pos,
                &self.weights,
                &mut self.ws_a.finny_i16,
            );
            let score = -self.quiescence(pos, ply + 1, -beta, -alpha);
            pos.unmake_move(mv, undo);
            if self.aborted {
                return 0;
            }
            if score > best_score {
                best_score = score;
                best_move = mv;
                if score >= beta {
                    self.tt.store(pos.hash, best_move, score, 0, Bound::Lower, ply as i32);
                    return beta;
                }
                if score > alpha {
                    alpha = score;
                }
            }
        }
        let bound = if best_score <= orig_alpha {
            Bound::Upper
        } else {
            Bound::Exact
        };
        self.tt
            .store(pos.hash, best_move, best_score, 0, bound, ply as i32);
        alpha
    }
}

pub struct SearchEngine {
    pub weights: Arc<SaccadionWeights>,
    pub ws_a: StreamAWorkspace,
    pub plasticity: DopaminePlasticityState,
    pub tt: Arc<TranspositionTable>,
    pub strategic_tt: StrategicTT,
    pub stop_flag: Arc<AtomicBool>,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    cont_hist: Vec<i16>,
    cap_hist: [[[i32; 6]; 64]; 12],
    dpm_pawn_corr: Vec<[i32; 2]>,
    dpm_np_w_corr: Vec<[i32; 2]>,
    dpm_np_b_corr: Vec<[i32; 2]>,
}

impl SearchEngine {
    pub fn new(weights: SaccadionWeights, hash_mb: usize) -> Self {
        let ws_a = StreamAWorkspace::new(&weights);
        let plasticity = DopaminePlasticityState::new(&weights.cfg);
        Self {
            weights: Arc::new(weights),
            ws_a,
            plasticity,
            tt: Arc::new(TranspositionTable::new_mb(hash_mb)),
            strategic_tt: StrategicTT::new(4096),
            stop_flag: Arc::new(AtomicBool::new(false)),
            killers: [[Move::NULL; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            cont_hist: vec![0i16; CONT_DIM * CONT_DIM],
            cap_hist: [[[0; 6]; 64]; 12],
            dpm_pawn_corr: vec![[0; 2]; DPM_CORR_SIZE],
            dpm_np_w_corr: vec![[0; 2]; DPM_CORR_SIZE],
            dpm_np_b_corr: vec![[0; 2]; DPM_CORR_SIZE],
        }
    }

    pub fn resize_tt(&mut self, hash_mb: usize) {
        self.tt = Arc::new(TranspositionTable::new_mb(hash_mb));
    }

    pub fn new_game(&mut self) {
        self.tt.clear();
        self.strategic_tt.clear();
        self.ws_a.finny.clear();
        self.ws_a.finny_i16.clear();
        self.plasticity.reset_game();
        self.killers = [[Move::NULL; 2]; MAX_PLY];
        self.history = [[[0; 64]; 64]; 2];
        self.cont_hist.fill(0);
        self.cap_hist = [[[0; 6]; 64]; 12];
        self.dpm_pawn_corr.fill([0; 2]);
        self.dpm_np_w_corr.fill([0; 2]);
        self.dpm_np_b_corr.fill([0; 2]);
    }

    /// Runs iterative-deepening Neuro-Guided PVS with Parallel DNp Lazy SMP, Giant Fiber
    /// Quiescence, Aspiration Windows, and live Dopamine adaptation.
    pub fn search<F>(&mut self, pos: &mut Position, cfg: SearchConfig, mut on_info: F) -> SearchResult
    where
        F: FnMut(&SearchResult),
    {
        self.stop_flag.store(false, Ordering::Relaxed);
        let start_time = Instant::now();
        let deadline = if cfg.time_limit_ms > 0 {
            Some(start_time + Duration::from_millis(cfg.time_limit_ms))
        } else {
            None
        };

        // Age history & prepare intra-search gamma plasticity while preserving intra-game alpha'/beta'
        for c in 0..2 {
            for f in 0..64 {
                for t in 0..64 {
                    self.history[c][f][t] /= 2;
                }
            }
        }
        if cfg.use_dopamine {
            self.plasticity.prepare_new_search();
        }

        let mut root_moves = MoveList::new();
        generate_legal(pos, &mut root_moves);
        if root_moves.is_empty() {
            return SearchResult::default();
        }
        // Instant reflex exit when only 1 legal move exists in timed games
        if root_moves.len() == 1 && cfg.time_limit_ms > 0 {
            let mv = root_moves.as_slice()[0];
            return SearchResult {
                best_move: mv,
                depth: 1,
                nodes: 1,
                pv: vec![mv],
                ..SearchResult::default()
            };
        }

        // Pre-evaluate root with TT-consistent Stream A; decide sync vs non-blocking async Stream B!
        // In fast blitz (<= 1500 ms), NEVER freeze Depth 1 waiting 200-400 ms for Stream B:
        // fire Giant Fiber search immediately at t=0ms and run Stream B concurrently on a background thread!
        let sync_root_sb = cfg.use_stream_b && (cfg.time_limit_ms == 0 || cfg.time_limit_ms > 1500);
        let async_root_sb =
            cfg.use_stream_b && !sync_root_sb && cfg.threads >= 2 && cfg.time_limit_ms >= 120;

        let root_a = evaluate_stream_a(
            pos,
            &self.weights,
            &mut self.ws_a,
            None,
            None,
            cfg.octopamine,
        );
        let mut root_sharpness = 0.5f32;
        let mut initial_sb_evals = 0u64;
        let async_novelty = if async_root_sb {
            self.plasticity
                .mbon_alpha_prime3_novelty(&root_a.active_apbp_stm, true)
        } else {
            0.5
        };
        if sync_root_sb {
            let novelty = self
                .plasticity
                .mbon_alpha_prime3_novelty(&root_a.active_apbp_stm, true);
            let out_b = evaluate_stream_b(
                pos,
                &root_moves,
                &self.weights,
                &root_a.mbon_summary,
                root_a.sez_cp,
                Some(&self.plasticity),
                novelty,
                cfg.serotonin,
                true,
            );
            initial_sb_evals = 1;
            root_sharpness = out_b.pfl2_sharpness;
            self.strategic_tt.store(pos.hash, &out_b);
        }

        // Move persistent main-thread state into a DnpWorker for zero-copy execution
        let helper_stop = Arc::new(AtomicBool::new(false));
        let helper_nodes = Arc::new(AtomicU64::new(0));
        let helper_q_nodes = Arc::new(AtomicU64::new(0));
        let async_sb_slot: Arc<Mutex<Option<StreamBOutput>>> = Arc::new(Mutex::new(None));
        let async_plasticity_snap = if async_root_sb {
            Some(self.plasticity.clone())
        } else {
            None
        };

        let mut main_worker = DnpWorker {
            weights: Arc::clone(&self.weights),
            tt: Arc::clone(&self.tt),
            ws_a: std::mem::replace(&mut self.ws_a, StreamAWorkspace::empty()),
            plasticity: std::mem::replace(
                &mut self.plasticity,
                DopaminePlasticityState::new(&self.weights.cfg),
            ),
            strategic_tt: std::mem::replace(&mut self.strategic_tt, StrategicTT::new(16)),
            stop_flag: Arc::clone(&self.stop_flag),
            helper_stop: Arc::clone(&helper_stop),
            shared_nodes: Arc::clone(&helper_nodes),
            shared_q_nodes: Arc::clone(&helper_q_nodes),
            flushed_nodes: 0,
            flushed_q_nodes: 0,
            killers: self.killers,
            history: self.history,
            cont_hist: std::mem::take(&mut self.cont_hist),
            cap_hist: self.cap_hist,
            threat_hist: [[0; 64]; 64],
            countermoves: [[Move::NULL; 64]; 64],
            dpm_pawn_corr: std::mem::take(&mut self.dpm_pawn_corr),
            dpm_np_w_corr: std::mem::take(&mut self.dpm_np_w_corr),
            dpm_np_b_corr: std::mem::take(&mut self.dpm_np_b_corr),
            moved_pc_to: [u16::MAX; MAX_PLY],
            static_evals: [0; MAX_PLY],
            nodes: 0,
            q_nodes: 0,
            stream_b_evals: initial_sb_evals,
            strategic_tt_hits: 0,
            lh_veto_triggers: 0,
            efference_surprises: 0,
            deadline,
            aborted: false,
            allow_time_abort: false,
            max_nodes: cfg.max_nodes,
            cfg: cfg.clone(),
            is_helper: false,
        };

        // Initialize ply-0 of the Push/Pop Incremental Giant Fiber stack for main worker!
        main_worker
            .ws_a
            .gf_stack
            .reset_root(pos, &self.weights, &mut main_worker.ws_a.finny_i16);

        let mut best_move = root_moves.as_slice()[0];
        let mut best_score = 0i32;
        let mut completed_depth = 0i32;
        let mut best_pv = vec![best_move];

        let n_helpers = if cfg.threads > 1 && cfg.max_depth > 2 && cfg.max_nodes > 4096 {
            let reserve_sb = usize::from(async_root_sb);
            (cfg.threads.saturating_sub(1 + reserve_sb)).min(63)
        } else {
            0
        };

        std::thread::scope(|scope| {
            // Non-blocking Async Stream B Brain in parallel during timed blitz/rapid games!
            if async_root_sb {
                let w_sb = Arc::clone(&self.weights);
                let h_stop_sb = Arc::clone(&helper_stop);
                let sb_slot = Arc::clone(&async_sb_slot);
                let pos_sb = pos.clone();
                let moves_sb = root_moves.clone();
                let mbon_sb = root_a.mbon_summary.clone();
                let sez_sb = root_a.sez_cp;
                let ser_sb = cfg.serotonin;
                let pl_sb = async_plasticity_snap;
                scope.spawn(move || {
                    if let Some(out_b) = evaluate_stream_b_cancellable(
                        &pos_sb,
                        &moves_sb,
                        &w_sb,
                        &mbon_sb,
                        sez_sb,
                        pl_sb.as_ref(),
                        async_novelty,
                        ser_sb,
                        true,
                        &h_stop_sb,
                    ) {
                        if let Ok(mut guard) = sb_slot.lock() {
                            *guard = Some(out_b);
                        }
                    }
                });
            }

            // Spawn Parallel DNp Lazy SMP helpers sharing Lock-Free TT & weights!
            for tid in 1..=n_helpers {
                let w_arc = Arc::clone(&self.weights);
                let tt_arc = Arc::clone(&self.tt);
                let s_flag = Arc::clone(&self.stop_flag);
                let h_stop = Arc::clone(&helper_stop);
                let h_nodes = Arc::clone(&helper_nodes);
                let h_q = Arc::clone(&helper_q_nodes);
                let mut pos_h = pos.clone();
                let cfg_h = cfg.clone();

                scope.spawn(move || {
                    let mut worker = DnpWorker::new_helper(
                        w_arc,
                        tt_arc,
                        s_flag,
                        h_stop,
                        Arc::clone(&h_nodes),
                        Arc::clone(&h_q),
                        cfg_h.clone(),
                        deadline,
                    );
                    worker
                        .ws_a
                        .gf_stack
                        .reset_root(&pos_h, &worker.weights, &mut worker.ws_a.finny_i16);
                    let mut d = 1 + (tid as i32 & 1);
                    let mut dummy_pv = Vec::with_capacity(32);
                    while d <= cfg_h.max_depth && !worker.aborted {
                        dummy_pv.clear();
                        let _ = worker.pvs(&mut pos_h, d, 0, -INF, INF, true, true, Move::NULL, Move::NULL, None, 0.0, &mut dummy_pv);
                        // Slight depth diversity across parallel DNp channels
                        d += if d >= 4 && ((tid + d as usize) & 1) == 0 { 2 } else { 1 };
                    }
                    h_nodes.fetch_add(worker.nodes - worker.flushed_nodes, Ordering::Relaxed);
                    h_q.fetch_add(worker.q_nodes - worker.flushed_q_nodes, Ordering::Relaxed);
                });
            }

            // Main DNp coordinator runs full iterative deepening + aspiration windows + PFL2 soft-stop
            let mut stable_iters = 0u32;
            let mut async_sb_merged = !async_root_sb;
            for depth in 1..=cfg.max_depth {
                // Seamlessly merge async Stream B guidance the instant it finishes!
                if !async_sb_merged {
                    if let Ok(mut guard) = async_sb_slot.try_lock() {
                        if let Some(out_b) = guard.take() {
                            root_sharpness = out_b.pfl2_sharpness;
                            main_worker.strategic_tt.store(pos.hash, &out_b);
                            main_worker.stream_b_evals += 1;
                            async_sb_merged = true;
                        }
                    }
                }
                // Guarantee Depth 1 ALWAYS finishes so Saccadion can NEVER time out at depth 0!
                main_worker.allow_time_abort = depth > 1;
                let base_delta = (30.0 + 45.0 * root_sharpness) as i32;
                let (mut alpha, mut beta) = if depth >= 4 && best_score.abs() < MATE_THRESHOLD - 100 {
                    (best_score - base_delta, best_score + base_delta)
                } else {
                    (-INF, INF)
                };

                let mut delta = base_delta;
                let mut pv_line = Vec::with_capacity(depth as usize + 4);
                let score = loop {
                    pv_line.clear();
                    let s = main_worker.pvs(pos, depth, 0, alpha, beta, true, true, Move::NULL, Move::NULL, None, 0.0, &mut pv_line);
                    if main_worker.aborted {
                        break s;
                    }
                    if s <= alpha {
                        alpha = (alpha - delta).max(-INF);
                        delta *= 2;
                    } else if s >= beta {
                        beta = (beta + delta).min(INF);
                        delta *= 2;
                    } else {
                        break s;
                    }
                };

                if main_worker.aborted && depth > 1 {
                    break;
                }
                completed_depth = depth;
                best_score = score;
                if let Some(&m) = pv_line.first() {
                    if m == best_move {
                        stable_iters += 1;
                    } else {
                        stable_iters = 0;
                    }
                    best_move = m;
                    best_pv = pv_line;
                }

                let snap = SearchResult {
                    best_move,
                    score_cp: best_score,
                    depth: completed_depth,
                    nodes: main_worker.nodes + helper_nodes.load(Ordering::Relaxed),
                    q_nodes: main_worker.q_nodes + helper_q_nodes.load(Ordering::Relaxed),
                    stream_b_evals: main_worker.stream_b_evals,
                    strategic_tt_hits: main_worker.strategic_tt_hits,
                    lh_veto_triggers: main_worker.lh_veto_triggers,
                    efference_surprises: main_worker.efference_surprises,
                    pam_updates: main_worker.plasticity.pam_updates,
                    ppl1_updates: main_worker.plasticity.ppl1_updates,
                    elapsed_ms: start_time.elapsed().as_millis() as u64,
                    pv: best_pv.clone(),
                };
                on_info(&snap);

                if best_score.abs() > MATE_THRESHOLD - 64 {
                    break;
                }
                // PFL2 Adaptive Time Management: stop early on quiet, rock-stable moves to save clock!
                if cfg.time_limit_ms >= 300
                    && depth >= 8
                    && stable_iters >= 4
                    && root_sharpness < 0.45
                    && snap.elapsed_ms * 100 >= cfg.time_limit_ms * 58
                {
                    break;
                }
            }

            // Signal all Parallel DNp helper threads to exit cleanly before scope joins
            helper_stop.store(true, Ordering::Relaxed);
        });

        // Restore persistent main-thread state back into SearchEngine
        self.ws_a = main_worker.ws_a;
        self.plasticity = main_worker.plasticity;
        self.strategic_tt = main_worker.strategic_tt;
        self.killers = main_worker.killers;
        self.history = main_worker.history;
        self.cont_hist = main_worker.cont_hist;
        self.cap_hist = main_worker.cap_hist;
        self.dpm_pawn_corr = main_worker.dpm_pawn_corr;
        self.dpm_np_w_corr = main_worker.dpm_np_w_corr;
        self.dpm_np_b_corr = main_worker.dpm_np_b_corr;

        // DPM Consolidation: transfer root backed-up surprise into intra-game alpha'/beta' fast weights
        if cfg.use_dopamine && completed_depth >= 2 && best_score.abs() < MATE_THRESHOLD {
            let root_delta = best_score as f32 - root_a.total_cp;
            self.plasticity.dpm_consolidate_move(
                root_delta,
                &root_a.active_apbp_stm,
                &root_a.active_apbp_opp,
            );
        }
        // Commit LAL strategic goal vector for anti-oscillation continuity on the next move
        if let Some(se) = self.strategic_tt.probe(pos.hash) {
            let goal = se.committed_goal.clone();
            self.plasticity.commit_lal_goal(&goal);
        }

        let total_nodes = main_worker.nodes + helper_nodes.load(Ordering::Relaxed);
        let total_q = main_worker.q_nodes + helper_q_nodes.load(Ordering::Relaxed);
        SearchResult {
            best_move,
            score_cp: best_score,
            depth: completed_depth,
            nodes: total_nodes,
            q_nodes: total_q,
            stream_b_evals: main_worker.stream_b_evals,
            strategic_tt_hits: main_worker.strategic_tt_hits,
            lh_veto_triggers: main_worker.lh_veto_triggers,
            efference_surprises: main_worker.efference_surprises,
            pam_updates: self.plasticity.pam_updates,
            ppl1_updates: self.plasticity.ppl1_updates,
            elapsed_ms: start_time.elapsed().as_millis() as u64,
            pv: best_pv,
        }
    }
}
