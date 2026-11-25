//! Neuro-Guided PVS Hybrid CPU Search Engine uniting all 9 FlyWire v783 circuits:
//! - Stack-fast PVS + GNG Quiescence running Stream A on ~98% of nodes
//! - Selective Stream B invocation on strategic crossroads (depth >= D_cx, PV, Veto, Surprise)
//! - PFL3 (99% contralateral) move ordering & LMR immunity for quiet strategic plans/sacrifices
//! - PFL2 (50/50 bilateral) dynamic depth extensions & pruning
//! - Lateral Horn 98.2%-I Veto disabling unsafe NMP/RFP and clamping fortress delusions
//! - Noduli Efference-Copy child surprise recovery
//! - Eligibility-Trace bounded PAM / 12x PPL1 Dopamine feedback + LAL anti-oscillation + Serotonin contempt

use crate::tt::{score_from_tt, Bound, StrategicEntry, StrategicTT, TranspositionTable, MATE_SCORE, MATE_THRESHOLD};
use saccadion_chess::movegen::{generate_legal, generate_legal_tactical};
use saccadion_chess::position::Position;
use saccadion_chess::types::{Move, MoveList};
use saccadion_nn::plasticity::DopaminePlasticityState;
use saccadion_nn::stream_a::{evaluate_stream_a, StreamAWorkspace};
use saccadion_nn::stream_b::evaluate_stream_b;
use saccadion_nn::weights::SaccadionWeights;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_PLY: usize = 128;
const INF: i32 = 32_000;

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub max_depth: i32,
    pub max_nodes: u64,
    pub time_limit_ms: u64,
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
            cx_min_depth: 4,
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

pub struct SearchEngine {
    pub weights: SaccadionWeights,
    pub ws_a: StreamAWorkspace,
    pub plasticity: DopaminePlasticityState,
    pub tt: TranspositionTable,
    pub strategic_tt: StrategicTT,
    pub stop_flag: Arc<AtomicBool>,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    // Per-search telemetry
    nodes: u64,
    q_nodes: u64,
    stream_b_evals: u64,
    strategic_tt_hits: u64,
    lh_veto_triggers: u64,
    efference_surprises: u64,
    start_time: Instant,
    deadline: Option<Instant>,
    aborted: bool,
    max_nodes: u64,
    cfg: SearchConfig,
}

impl SearchEngine {
    pub fn new(weights: SaccadionWeights, hash_mb: usize) -> Self {
        let ws_a = StreamAWorkspace::new(&weights);
        let plasticity = DopaminePlasticityState::new(&weights.cfg);
        Self {
            weights,
            ws_a,
            plasticity,
            tt: TranspositionTable::new_mb(hash_mb),
            strategic_tt: StrategicTT::new(4096),
            stop_flag: Arc::new(AtomicBool::new(false)),
            killers: [[Move::NULL; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            nodes: 0,
            q_nodes: 0,
            stream_b_evals: 0,
            strategic_tt_hits: 0,
            lh_veto_triggers: 0,
            efference_surprises: 0,
            start_time: Instant::now(),
            deadline: None,
            aborted: false,
            max_nodes: u64::MAX,
            cfg: SearchConfig::default(),
        }
    }

    pub fn new_game(&mut self) {
        self.tt.clear();
        self.strategic_tt.clear();
        self.ws_a.finny.clear();
        self.plasticity.reset_game();
        self.killers = [[Move::NULL; 2]; MAX_PLY];
        self.history = [[[0; 64]; 64]; 2];
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
            if self.stop_flag.load(Ordering::Relaxed) {
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

    /// Runs iterative-deepening Neuro-Guided PVS with Aspiration Windows and live Dopamine adaptation.
    pub fn search<F>(&mut self, pos: &mut Position, cfg: SearchConfig, mut on_info: F) -> SearchResult
    where
        F: FnMut(&SearchResult),
    {
        self.cfg = cfg.clone();
        self.nodes = 0;
        self.q_nodes = 0;
        self.stream_b_evals = 0;
        self.strategic_tt_hits = 0;
        self.lh_veto_triggers = 0;
        self.efference_surprises = 0;
        self.aborted = false;
        self.max_nodes = cfg.max_nodes;
        self.stop_flag.store(false, Ordering::Relaxed);
        self.start_time = Instant::now();
        self.deadline = if cfg.time_limit_ms > 0 {
            Some(self.start_time + Duration::from_millis(cfg.time_limit_ms))
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
        if cfg.use_stream_b {
            let novelty = self.plasticity.mbon_alpha_prime3_novelty(&root_a.active_apbp_stm, true);
            let out_b = evaluate_stream_b(
                pos,
                &root_moves,
                &self.weights,
                &root_a.mbon_summary,
                Some(&self.plasticity),
                novelty,
                cfg.serotonin,
                true,
            );
            self.stream_b_evals += 1;
            root_sharpness = out_b.pfl2_sharpness;
            if cfg.use_dopamine {
                self.plasticity.inject_stream_b_seed(&out_b.gamma_seed, &out_b.gamma_seed, 0.25);
            }
            self.strategic_tt.store(pos.hash, &out_b);
        }

        let mut best_move = root_moves.as_slice()[0];
        let mut best_score = 0i32;
        let mut completed_depth = 0i32;
        let mut best_pv = vec![best_move];

        for depth in 1..=cfg.max_depth {
            // Aspiration window scaled dynamically by PFL2 sharpness!
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
                let s = self.pvs(pos, depth, 0, alpha, beta, true, None, 0.0, &mut pv_line);
                if self.aborted {
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

            if self.aborted && depth > 1 {
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
                nodes: self.nodes,
                q_nodes: self.q_nodes,
                stream_b_evals: self.stream_b_evals,
                strategic_tt_hits: self.strategic_tt_hits,
                lh_veto_triggers: self.lh_veto_triggers,
                efference_surprises: self.efference_surprises,
                pam_updates: self.plasticity.pam_updates,
                ppl1_updates: self.plasticity.ppl1_updates,
                elapsed_ms: self.start_time.elapsed().as_millis() as u64,
                pv: best_pv.clone(),
            };
            on_info(&snap);

            if best_score.abs() > MATE_THRESHOLD - 64 {
                break;
            }
        }

        // DPM Consolidation: transfer root backed-up surprise into intra-game alpha'/beta' fast weights
        if cfg.use_dopamine && completed_depth >= 2 && best_score.abs() < MATE_THRESHOLD {
            let root_delta = best_score as f32 - root_a.total_cp;
            self.plasticity
                .dpm_consolidate_move(root_delta, &root_a.active_apbp_stm, &root_a.active_apbp_opp);
        }
        // Commit LAL strategic goal vector for anti-oscillation continuity on the next move
        if let Some(se) = self.strategic_tt.probe(pos.hash) {
            let goal = se.committed_goal.clone();
            self.plasticity.commit_lal_goal(&goal);
        }

        SearchResult {
            best_move,
            score_cp: best_score,
            depth: completed_depth,
            nodes: self.nodes,
            q_nodes: self.q_nodes,
            stream_b_evals: self.stream_b_evals,
            strategic_tt_hits: self.strategic_tt_hits,
            lh_veto_triggers: self.lh_veto_triggers,
            efference_surprises: self.efference_surprises,
            pam_updates: self.plasticity.pam_updates,
            ppl1_updates: self.plasticity.ppl1_updates,
            elapsed_ms: self.start_time.elapsed().as_millis() as u64,
            pv: best_pv,
        }
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
            return self.eval_leaf_cp(pos, parent_cent_mask);
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
            return self.quiescence(pos, ply, alpha, beta, parent_cent_mask);
        }

        self.nodes += 1;

        // 1. Probe standard Transposition Table
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

        // 2. Fast Stream A Leaf Evaluation + Lateral Horn Innate Veto
        let out_a = evaluate_stream_a(
            pos,
            &self.weights,
            &mut self.ws_a,
            if self.cfg.use_dopamine { Some(&self.plasticity) } else { None },
            parent_cent_mask,
            self.cfg.octopamine,
        );
        let static_eval = out_a.total_cp.round() as i32;

        // Accumulate decaying eligibility trace on search stack for retroactive Dopamine credit
        // and trigger MBON -> DAN Second-Order Conditioning on stable alpha/beta valuation shifts
        if self.cfg.use_dopamine {
            self.plasticity.apply_mbon_soc_feedback(out_a.slow_cp);
            self.plasticity
                .accumulate_eligibility(&out_a.active_gamma_stm, &out_a.active_gamma_opp);
        }

        let lh_tactical_alert = self.cfg.use_lateral_horn && out_a.veto.tactical_alert();
        if lh_tactical_alert {
            self.lh_veto_triggers += 1;
        }

        // Noduli Efference-Copy Surprise Check
        let efference_surprise = self.cfg.use_efference_copy
            && ply > 0
            && expected_eval_from_parent != 0.0
            && (out_a.total_cp - expected_eval_from_parent).abs() > 140.0;
        if efference_surprise {
            self.efference_surprises += 1;
        }

        // 3. Reverse Futility Pruning (RFP) & Null-Move Pruning (NMP) — strictly vetoed by Lateral Horn!
        if !is_pv && !in_check && !lh_tactical_alert && !efference_surprise {
            // Reverse Futility Pruning
            if depth <= 3 && static_eval - 95 * depth >= beta {
                return static_eval;
            }
            // Null-Move Pruning (ensure side to move has non-pawn material to avoid Zugzwang)
            let us = pos.side_to_move;
            let non_pawn = pos.color_occ[us.index()]
                ^ pos.piece_bb(us, saccadion_chess::types::Piece::Pawn)
                ^ pos.piece_bb(us, saccadion_chess::types::Piece::King);
            if depth >= 3 && static_eval >= beta && non_pawn.non_empty() {
                let r = 2 + depth / 4;
                let undo = pos.make_null_move();
                let mut dummy_pv = Vec::new();
                let null_score =
                    -self.pvs(pos, depth - 1 - r, ply + 1, -beta, -beta + 1, false, parent_cent_mask, 0.0, &mut dummy_pv);
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

        // 5. Selective Stream B Central Complex Oracle (only ~1-2% of nodes, cached in StrategicTT!)
        let need_stream_b = self.cfg.use_stream_b
            && (depth >= self.cfg.cx_min_depth || (is_pv && depth >= 2) || efference_surprise);

        let mut local_strat: Option<StrategicEntry> = None;
        if need_stream_b {
            if let Some(hit) = self.strategic_tt.probe(pos.hash) {
                self.strategic_tt_hits += 1;
                local_strat = Some(hit.clone());
            } else {
                let novelty = self.plasticity.mbon_alpha_prime3_novelty(&out_a.active_apbp_stm, false);
                let out_b = evaluate_stream_b(
                    pos,
                    &moves,
                    &self.weights,
                    &out_a.mbon_summary,
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

        // Top-down Centrifugal KC mask inherited by children
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
                -(out_a.total_cp + noduli_delta)
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
                // Late Move Reductions (LMR) modulated by PFL3 Immunity & PFL2 Sharpness!
                let mut r = 0i32;
                if depth >= 3 && move_idx >= 3 && is_quiet && !in_check && !gives_check {
                    r = 1 + (depth / 4) + (move_idx as i32 / 8);
                    // PFL3 Top-3 strategic quiet moves get LMR Immunity!
                    if pfl3_rank < 3 {
                        r = (r - 3).max(0);
                    }
                    // Calm low-PFL2 positions prune harder to save CPU NPS
                    if pfl2_sharpness < 0.30 {
                        r += 1;
                    }
                    if lh_tactical_alert || efference_surprise {
                        r = (r - 1).max(0);
                    }
                    r = r.min(depth - 2).max(0);
                }

                // Zero-window scout search
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
                // Re-search at full depth if reduced move beat alpha
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
                // Full-window PV re-search
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

        // 7. Live Bounded PAM / 12x PPL1 Dopamine Update on meaningful backed-up surprises!
        if self.cfg.use_dopamine && depth >= 3 && best_score.abs() < MATE_THRESHOLD {
            let delta_cp = best_score as f32 - out_a.total_cp;
            self.plasticity.apply_dopamine_surprise(
                delta_cp,
                &out_a.active_gamma_stm,
                &out_a.active_gamma_opp,
            );
        }

        // 8. Store in Transposition Table
        let bound = if best_score <= orig_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt.store(pos.hash, best_move, best_score, depth, bound, ply as i32);

        best_score
    }

    /// GNG Tactical Quiescence Search (captures, promotions, and check evasions).
    fn quiescence(
        &mut self,
        pos: &mut Position,
        ply: usize,
        mut alpha: i32,
        beta: i32,
        cent_mask: Option<&[f32]>,
    ) -> i32 {
        if self.check_stop() {
            return 0;
        }
        self.nodes += 1;
        self.q_nodes += 1;
        if ply >= MAX_PLY - 1 {
            return self.eval_leaf_cp(pos, cent_mask);
        }

        let in_check = pos.in_check();
        let stand_pat = self.eval_leaf_cp(pos, cent_mask);
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

        // MVV-LVA tactical ordering
        let mut scored: Vec<(Move, i32)> = moves
            .as_slice()
            .iter()
            .map(|&mv| {
                let vic = pos.piece_at(mv.to_sq()).map(|(_, p)| p.cp_value()).unwrap_or(100);
                let atk = pos.piece_at(mv.from_sq()).map(|(_, p)| p.cp_value()).unwrap_or(100);
                (mv, vic * 10 - atk)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        for (mv, _) in scored {
            // SEE pruning for losing captures when not in check
            if !in_check && mv.is_capture() && !mv.is_promotion() && !pos.see_ge(mv, -50) {
                continue;
            }
            let undo = pos.make_move(mv);
            let score = -self.quiescence(pos, ply + 1, -beta, -alpha, cent_mask);
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

    #[inline]
    fn eval_leaf_cp(&mut self, pos: &Position, cent_mask: Option<&[f32]>) -> i32 {
        let out = evaluate_stream_a(
            pos,
            &self.weights,
            &mut self.ws_a,
            if self.cfg.use_dopamine { Some(&self.plasticity) } else { None },
            cent_mask,
            self.cfg.octopamine,
        );
        out.total_cp.round() as i32
    }
}
