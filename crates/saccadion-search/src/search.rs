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
use saccadion_chess::types::{Move, MoveList};
use saccadion_nn::plasticity::DopaminePlasticityState;
use saccadion_nn::stream_a::{evaluate_giant_fiber_cp, evaluate_stream_a, StreamAWorkspace};
use saccadion_nn::stream_b::evaluate_stream_b;
use saccadion_nn::weights::SaccadionWeights;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_PLY: usize = 128;
const INF: i32 = 32_000;

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
            cx_min_depth: 7,
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
    nodes: u64,
    q_nodes: u64,
    stream_b_evals: u64,
    strategic_tt_hits: u64,
    lh_veto_triggers: u64,
    efference_surprises: u64,
    deadline: Option<Instant>,
    aborted: bool,
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
        gamma_seed: Option<&[f32]>,
    ) -> Self {
        let ws_a = StreamAWorkspace::new(&weights);
        let mut plasticity = DopaminePlasticityState::new(&weights.cfg);
        if cfg.use_dopamine {
            if let Some(seed) = gamma_seed {
                plasticity.inject_stream_b_seed(seed, seed, 0.25);
            }
        }
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
            nodes: 0,
            q_nodes: 0,
            stream_b_evals: 0,
            strategic_tt_hits: 0,
            lh_veto_triggers: 0,
            efference_surprises: 0,
            deadline,
            aborted: false,
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
            if let Some(dl) = self.deadline {
                if Instant::now() >= dl {
                    self.aborted = true;
                    return true;
                }
            }
        }
        false
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
        parent_cent_mask: Option<&[f32]>,
        expected_eval_from_parent: f32,
        pv_out: &mut Vec<Move>,
    ) -> i32 {
        if self.check_stop() {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return evaluate_giant_fiber_cp(pos, &self.weights, &mut self.ws_a);
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

        // 1. Probe Lock-Free Atomic Transposition Table
        let mut tt_move = Move::NULL;
        if let Some(entry) = self.tt.probe(pos.hash) {
            tt_move = entry.best_move;
            if !is_pv && entry.depth as i32 >= depth {
                let tt_score = score_from_tt(entry.score as i32, ply as i32);
                match entry.bound {
                    Bound::Exact => return tt_score,
                    Bound::Lower if tt_score >= beta => return tt_score,
                    Bound::Upper if tt_score <= alpha => return tt_score,
                    _ => {}
                }
            }
        }

        // 2. Biological 3-Tier Evaluation Hierarchy:
        //    - Tier 1 (Shallow Non-PV Frontier depth <= 2): Giant Fiber (SEZ + alpha/beta MBON)
        //      delivers 10x-20x higher CPU NPS without rebuilding full 64-sq rays/claws!
        //    - Tier 2 (PV or depth >= 3): Full Stream A (FullRays + FlyHash + Lateral Horn + Dopamine)
        let cached_strat_hit = if self.cfg.use_stream_b && (is_pv || depth >= 3) {
            self.strategic_tt.probe(pos.hash).cloned()
        } else {
            None
        };

        let use_fast_frontier = !is_pv && depth <= 2 && cached_strat_hit.is_none();
        let (static_eval, out_a_opt, lh_tactical_alert, efference_surprise) = if use_fast_frontier {
            let gf_cp = evaluate_giant_fiber_cp(pos, &self.weights, &mut self.ws_a);
            (gf_cp, None, false, false)
        } else {
            let out_a = evaluate_stream_a(
                pos,
                &self.weights,
                &mut self.ws_a,
                if self.cfg.use_dopamine { Some(&self.plasticity) } else { None },
                parent_cent_mask,
                self.cfg.octopamine,
            );
            let se = if let Some(ref hit) = cached_strat_hit {
                (0.75 * hit.deep_cp + 0.25 * out_a.total_cp).round() as i32
            } else {
                out_a.total_cp.round() as i32
            };
            if self.cfg.use_dopamine {
                self.plasticity.apply_mbon_soc_feedback(out_a.slow_cp);
                self.plasticity
                    .accumulate_eligibility(&out_a.active_gamma_stm, &out_a.active_gamma_opp);
            }
            let lh_alert = self.cfg.use_lateral_horn && out_a.veto.tactical_alert();
            if lh_alert {
                self.lh_veto_triggers += 1;
            }
            let eff_surp = self.cfg.use_efference_copy
                && ply > 0
                && expected_eval_from_parent != 0.0
                && (out_a.total_cp - expected_eval_from_parent).abs() > 140.0;
            if eff_surp {
                self.efference_surprises += 1;
            }
            (se, Some(out_a), lh_alert, eff_surp)
        };
        let parent_eval_f32 = out_a_opt
            .as_ref()
            .map(|o| o.total_cp)
            .unwrap_or(static_eval as f32);

        // 3. Reverse Futility Pruning (RFP) & Null-Move Pruning (NMP) — strictly vetoed by Lateral Horn!
        if !is_pv && !in_check && !lh_tactical_alert && !efference_surprise {
            if depth <= 3 && static_eval - 95 * depth >= beta {
                return static_eval;
            }
            let us = pos.side_to_move;
            let non_pawn = pos.color_occ[us.index()]
                ^ pos.piece_bb(us, saccadion_chess::types::Piece::Pawn)
                ^ pos.piece_bb(us, saccadion_chess::types::Piece::King);
            if depth >= 3 && static_eval >= beta && non_pawn.non_empty() {
                let r = 2 + depth / 4;
                let undo = pos.make_null_move();
                let mut dummy_pv = Vec::new();
                let null_score = -self.pvs(
                    pos,
                    depth - 1 - r,
                    ply + 1,
                    -beta,
                    -beta + 1,
                    false,
                    parent_cent_mask,
                    0.0,
                    &mut dummy_pv,
                );
                pos.unmake_null_move(undo);
                if self.aborted {
                    return 0;
                }
                if null_score >= beta && null_score < MATE_THRESHOLD {
                    return beta;
                }
            }
        }

        // 4. Generate legal moves & handle Checkmate / Stalemate
        let mut moves = MoveList::new();
        generate_legal(pos, &mut moves);
        if moves.is_empty() {
            return if in_check { -MATE_SCORE + ply as i32 } else { 0 };
        }

        // 5. Selective Stream B Central Complex Oracle (main thread & high-depth crossroads only!)
        let need_stream_b = self.cfg.use_stream_b
            && !self.is_helper
            && out_a_opt.is_some()
            && (depth >= self.cfg.cx_min_depth
                || (is_pv && depth >= 6)
                || (efference_surprise && depth >= 5));

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

        // PFL2 Sharpness-driven depth extension on critical turning points
        let pfl2_sharpness = local_strat.as_ref().map(|s| s.pfl2_sharpness).unwrap_or(0.5);
        if is_pv && pfl2_sharpness > 0.78 && !in_check && ply < 16 && depth >= 3 {
            depth += 1;
        }

        // 6. Move Ordering: TT Move > PFL3 Goal-Aligned Policy > MVV-LVA Captures > Killers > History
        let mut scored_moves: Vec<(Move, i32, usize, f32)> = Vec::with_capacity(moves.len());
        let stm_idx = pos.side_to_move.index();
        for &mv in moves.as_slice() {
            let mut priority = 0i32;
            let mut pfl3_rank = usize::MAX;
            let mut noduli_delta = 0.0f32;

            if mv == tt_move {
                priority += 2_000_000;
            }
            if let Some(ref st) = local_strat {
                if let Some((rank, pm)) = st.top_moves.iter().enumerate().find(|(_, m)| m.mv == mv) {
                    pfl3_rank = rank;
                    noduli_delta = pm.expected_delta_cp;
                    priority += 450_000 + (pm.prob * 500_000.0) as i32;
                }
            }
            if mv.is_capture() {
                let vic = pos
                    .piece_at(mv.to_sq())
                    .map(|(_, p)| p.cp_value())
                    .unwrap_or(100);
                let atk = pos
                    .piece_at(mv.from_sq())
                    .map(|(_, p)| p.cp_value())
                    .unwrap_or(100);
                priority += 1_000_000 + vic * 10 - atk;
            } else if mv.is_promotion() {
                priority += 950_000;
            } else if mv == self.killers[ply][0] {
                priority += 400_000;
            } else if mv == self.killers[ply][1] {
                priority += 350_000;
            } else {
                priority += self.history[stm_idx][mv.from_sq().index()][mv.to_sq().index()];
            }
            scored_moves.push((mv, priority, pfl3_rank, noduli_delta));
        }
        scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        let child_cent_mask: Option<&[f32]> = local_strat
            .as_ref()
            .map(|s| s.centrifugal_kc_mask.as_slice())
            .or(parent_cent_mask);

        let orig_alpha = alpha;
        let mut best_move = scored_moves[0].0;
        let mut best_score = -INF;
        let mut child_pv = Vec::with_capacity(depth as usize);

        for (move_idx, &(mv, _, pfl3_rank, noduli_delta)) in scored_moves.iter().enumerate() {
            let is_quiet = !mv.is_capture() && !mv.is_promotion();
            let undo = pos.make_move(mv);
            let gives_check = pos.in_check();
            let expected_child_eval = if noduli_delta != 0.0 {
                -(parent_eval_f32 + noduli_delta)
            } else {
                0.0
            };

            child_pv.clear();
            let score = if move_idx == 0 {
                -self.pvs(
                    pos,
                    depth - 1,
                    ply + 1,
                    -beta,
                    -alpha,
                    is_pv,
                    child_cent_mask,
                    expected_child_eval,
                    &mut child_pv,
                )
            } else {
                let mut r = 0i32;
                if depth >= 3 && move_idx >= 3 && is_quiet && !in_check && !gives_check {
                    r = 1 + (depth / 4) + (move_idx as i32 / 8);
                    if pfl3_rank < 3 {
                        r = (r - 3).max(0);
                    }
                    if pfl2_sharpness < 0.30 {
                        r += 1;
                    }
                    if lh_tactical_alert || efference_surprise {
                        r = (r - 1).max(0);
                    }
                    r = r.min(depth - 2).max(0);
                }

                let mut s = -self.pvs(
                    pos,
                    depth - 1 - r,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                    false,
                    child_cent_mask,
                    expected_child_eval,
                    &mut child_pv,
                );
                if s > alpha && r > 0 {
                    child_pv.clear();
                    s = -self.pvs(
                        pos,
                        depth - 1,
                        ply + 1,
                        -alpha - 1,
                        -alpha,
                        false,
                        child_cent_mask,
                        expected_child_eval,
                        &mut child_pv,
                    );
                }
                if s > alpha && s < beta {
                    child_pv.clear();
                    s = -self.pvs(
                        pos,
                        depth - 1,
                        ply + 1,
                        -beta,
                        -alpha,
                        true,
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
                        if is_quiet {
                            self.killers[ply][1] = self.killers[ply][0];
                            self.killers[ply][0] = mv;
                            let h = &mut self.history[stm_idx][mv.from_sq().index()][mv.to_sq().index()];
                            *h = (*h + depth * depth).min(200_000);
                        }
                        break;
                    }
                }
            }
        }

        // 7. Live Bounded PAM / 12x PPL1 / Dop1R2 Dopamine Update
        if self.cfg.use_dopamine && depth >= 3 && best_score.abs() < MATE_THRESHOLD {
            if let Some(ref out_a) = out_a_opt {
                let delta_cp = best_score as f32 - out_a.total_cp;
                self.plasticity.apply_dopamine_surprise(
                    delta_cp,
                    &out_a.active_gamma_stm,
                    &out_a.active_gamma_opp,
                );
            }
        }

        // 8. Store in Lock-Free Atomic Transposition Table
        let bound = if best_score <= orig_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt
            .store(pos.hash, best_move, best_score, depth, bound, ply as i32);

        best_score
    }

    /// LC4/LPLC2 -> Giant Fiber (GF / DNp01) Ultra-Fast Tactical Quiescence Reflex!
    ///
    /// Evaluates looming captures, promotions, and check evasions using incremental FinnyCache
    /// SEZ Innate Survival Valence + frozen alpha/beta MBONs (<1.5 us -> 20x-40x higher NPS!).
    fn quiescence(&mut self, pos: &mut Position, ply: usize, mut alpha: i32, beta: i32) -> i32 {
        if self.check_stop() {
            return 0;
        }
        self.nodes += 1;
        self.q_nodes += 1;
        // QSearch stand-pat uses ultra-fast SEZ-only Giant Fiber (~150ns = millions NPS/core!)
        // while interior PVS nodes blend SEZ + alpha/beta MBON + Full Stream A + Stream B!
        if ply >= MAX_PLY - 1 {
            return self.ws_a.finny.compute_sez_only_cp(pos, &self.weights).round() as i32;
        }

        let in_check = pos.in_check();
        let stand_pat = self.ws_a.finny.compute_sez_only_cp(pos, &self.weights).round() as i32;
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
                // Delta pruning: skip captures that cannot possibly lift score to alpha
                if stand_pat + vic_cp + 220 < alpha {
                    continue;
                }
                // SEE pruning for clearly losing captures
                if !pos.see_ge(mv, -50) {
                    continue;
                }
            }
            let undo = pos.make_move(mv);
            let score = -self.quiescence(pos, ply + 1, -beta, -alpha);
            pos.unmake_move(mv, undo);
            if self.aborted {
                return 0;
            }
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
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
        }
    }

    pub fn resize_tt(&mut self, hash_mb: usize) {
        self.tt = Arc::new(TranspositionTable::new_mb(hash_mb));
    }

    pub fn new_game(&mut self) {
        self.tt.clear();
        self.strategic_tt.clear();
        self.ws_a.finny.clear();
        self.plasticity.reset_game();
        self.killers = [[Move::NULL; 2]; MAX_PLY];
        self.history = [[[0; 64]; 64]; 2];
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

        // Pre-evaluate root with Stream A + Stream B for top-down centrifugal/seed steering
        let root_a = evaluate_stream_a(
            pos,
            &self.weights,
            &mut self.ws_a,
            if cfg.use_dopamine { Some(&self.plasticity) } else { None },
            None,
            cfg.octopamine,
        );
        let mut root_sharpness = 0.5f32;
        let mut root_gamma_seed: Option<Vec<f32>> = None;
        let mut initial_sb_evals = 0u64;
        if cfg.use_stream_b {
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
            if cfg.use_dopamine {
                self.plasticity
                    .inject_stream_b_seed(&out_b.gamma_seed, &out_b.gamma_seed, 0.25);
                root_gamma_seed = Some(out_b.gamma_seed.clone());
            }
            self.strategic_tt.store(pos.hash, &out_b);
        }

        // Move persistent main-thread state into a DnpWorker for zero-copy execution
        let helper_stop = Arc::new(AtomicBool::new(false));
        let helper_nodes = Arc::new(AtomicU64::new(0));
        let helper_q_nodes = Arc::new(AtomicU64::new(0));

        let mut main_worker = DnpWorker {
            weights: Arc::clone(&self.weights),
            tt: Arc::clone(&self.tt),
            ws_a: std::mem::replace(&mut self.ws_a, StreamAWorkspace::new(&self.weights)),
            plasticity: std::mem::replace(
                &mut self.plasticity,
                DopaminePlasticityState::new(&self.weights.cfg),
            ),
            strategic_tt: std::mem::replace(&mut self.strategic_tt, StrategicTT::new(256)),
            stop_flag: Arc::clone(&self.stop_flag),
            helper_stop: Arc::clone(&helper_stop),
            shared_nodes: Arc::clone(&helper_nodes),
            shared_q_nodes: Arc::clone(&helper_q_nodes),
            flushed_nodes: 0,
            flushed_q_nodes: 0,
            killers: self.killers,
            history: self.history,
            nodes: 0,
            q_nodes: 0,
            stream_b_evals: initial_sb_evals,
            strategic_tt_hits: 0,
            lh_veto_triggers: 0,
            efference_surprises: 0,
            deadline,
            aborted: false,
            max_nodes: cfg.max_nodes,
            cfg: cfg.clone(),
            is_helper: false,
        };

        let mut best_move = root_moves.as_slice()[0];
        let mut best_score = 0i32;
        let mut completed_depth = 0i32;
        let mut best_pv = vec![best_move];

        let n_helpers = if cfg.threads > 1 && cfg.max_depth > 2 && cfg.max_nodes > 4096 {
            (cfg.threads - 1).min(63)
        } else {
            0
        };

        std::thread::scope(|scope| {
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
                let seed_ref = root_gamma_seed.as_deref();

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
                        seed_ref,
                    );
                    let mut d = 1 + (tid as i32 & 1);
                    let mut dummy_pv = Vec::with_capacity(32);
                    while d <= cfg_h.max_depth && !worker.aborted {
                        dummy_pv.clear();
                        let _ = worker.pvs(&mut pos_h, d, 0, -INF, INF, true, None, 0.0, &mut dummy_pv);
                        // Slight depth diversity across parallel DNp channels
                        d += if d >= 4 && ((tid + d as usize) & 1) == 0 { 2 } else { 1 };
                    }
                    h_nodes.fetch_add(worker.nodes - worker.flushed_nodes, Ordering::Relaxed);
                    h_q.fetch_add(worker.q_nodes - worker.flushed_q_nodes, Ordering::Relaxed);
                });
            }

            // Main DNp coordinator runs full iterative deepening + aspiration windows + UCI telemetry
            for depth in 1..=cfg.max_depth {
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
                    let s = main_worker.pvs(pos, depth, 0, alpha, beta, true, None, 0.0, &mut pv_line);
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
