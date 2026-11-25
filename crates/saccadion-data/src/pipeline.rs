//! Beyond-Stockfish game-to-shard trajectory pipeline: extracts 10-20 ply Strategic Pressure,
//! Lateral Horn targets, ThreatMap64, 2-ply VNC follow-ups, and K=12 trajectory windows.

use crate::format::SaccRecord;
use crate::pgn::ParsedGame;
use saccadion_chess::features::{extract_lateral_horn_features, extract_threat_masks};
use saccadion_chess::position::Position;
use saccadion_chess::pressure::{compute_pressure_snapshot, compute_trajectory_pressure};
use saccadion_chess::types::Color;

fn cp_to_wdl_probs(cp: f32) -> [f32; 3] {
    let win = 1.0 / (1.0 + (-(cp - 90.0) / 140.0).exp());
    let loss = 1.0 / (1.0 + ((cp + 90.0) / 140.0).exp());
    let draw = (1.0 - win - loss).max(0.02);
    let s = win + draw + loss;
    [win / s, draw / s, loss / s]
}

/// Converts a ParsedGame into dense multi-sensor SaccRecords with exact 10-20 ply Strategic Pressure.
pub fn game_to_sacc_records(game: &ParsedGame) -> Vec<SaccRecord> {
    let n_moves = game.moves.len();
    if n_moves < 6 {
        return Vec::new();
    }

    // Pass 1: Replay full game (respecting [FEN] header for Fishtest/opening-book games)
    let mut pos = match &game.start_fen {
        Some(fen) => Position::from_fen(fen).unwrap_or_else(|_| Position::startpos()),
        None => Position::startpos(),
    };
    let mut positions = Vec::with_capacity(n_moves);
    let mut snapshots = Vec::with_capacity(n_moves);
    let mut stm_colors = Vec::with_capacity(n_moves);
    let mut evals_stm = Vec::with_capacity(n_moves);

    for &(mv, eval_stm_opt) in &game.moves {
        let stm = pos.side_to_move;
        let snap = compute_pressure_snapshot(&pos);
        let eval_stm = eval_stm_opt.unwrap_or_else(|| pos.material_balance_stm() as f32);

        positions.push(pos.clone());
        snapshots.push(snap);
        stm_colors.push(stm);
        evals_stm.push(eval_stm);
        pos.make_move(mv);
    }

    // Pass 2: Build rich 128-byte SaccRecords peeking +2, +6, +12, +20 plies ahead
    let mut records = Vec::with_capacity(n_moves);
    for ply in 0..n_moves {
        let p = &positions[ply];
        let stm = stm_colors[ply];
        let (board_nibbles, meta_flags) = SaccRecord::pack_board(p);

        // Soft WDL blending teacher eval (60%) with true long-horizon game outcome (40%)
        let outcome_stm = if stm == Color::White {
            game.white_result
        } else {
            1.0 - game.white_result
        };
        let out_wdl = if outcome_stm > 0.75 {
            [0.94, 0.04, 0.02]
        } else if outcome_stm < 0.25 {
            [0.02, 0.04, 0.94]
        } else {
            [0.08, 0.84, 0.08]
        };
        let eval_wdl = cp_to_wdl_probs(evals_stm[ply]);
        let has_engine_eval = game.moves[ply].1.is_some();
        // Engine-evaluated Fishtest/Lichess positions carry +/-10 cp ground truth (85% eval / 15% outcome!)
        let (w_ev, w_out) = if has_engine_eval { (0.85, 0.15) } else { (0.20, 0.80) };
        let wdl_target = [
            w_ev * eval_wdl[0] + w_out * out_wdl[0],
            w_ev * eval_wdl[1] + w_out * out_wdl[1],
            w_ev * eval_wdl[2] + w_out * out_wdl[2],
        ];

        let best_mv = game.moves[ply].0;
        let best_u16 = best_mv.from_to_index() as u16;
        // 2-ply VNC follow-up move (our next move after opponent replies)
        let next_u16 = if ply + 2 < n_moves {
            game.moves[ply + 2].0.from_to_index() as u16
        } else {
            best_u16
        };

        let pressure = compute_trajectory_pressure(&snapshots, &stm_colors, ply);
        let lh_raw = extract_lateral_horn_features(p);
        let lh_targets = [
            lh_raw[8],                        // blunder_risk (hanging own high-value piece)
            snapshots[ply].fortress_index,    // fortress_prob
            (lh_raw[0] * lh_raw[4]).min(1.0), // mate_danger (check * exposed king ring)
            lh_raw[12].max(lh_raw[13]),       // passer_urgency
        ];

        // Tactical volatility: swing between current eval and +4 ply eval (trains PFL2!)
        let fut_eval = if ply + 4 < n_moves {
            evals_stm[ply + 4]
        } else {
            evals_stm[ply]
        };
        let volatility = ((fut_eval - evals_stm[ply]).abs() / 300.0).clamp(0.0, 1.0);
        let mlh = ((n_moves - ply) as f32 / 120.0).clamp(0.0, 1.0);
        let (threat_mask_stm, threat_mask_opp) = extract_threat_masks(p);

        records.push(SaccRecord {
            board_nibbles,
            meta_flags,
            wdl_target,
            best_move_u16: best_u16,
            next_move_u16: next_u16,
            top4_moves: [best_u16, 0, 0, 0],
            top4_probs_u16: [65535, 0, 0, 0],
            pressure_targets: pressure.values,
            lh_targets,
            volatility,
            mlh,
            threat_mask_stm,
            threat_mask_opp,
        });
    }

    records
}

/// Slices a game's records into contiguous K-step trajectory windows for Stage-3 Meta-Dopamine training.
pub fn extract_trajectory_windows(records: &[SaccRecord], window_k: usize) -> Vec<SaccRecord> {
    if records.len() < window_k {
        return Vec::new();
    }
    let mut out = Vec::new();
    let stride = (window_k / 2).max(1);
    let mut start = 0usize;
    while start + window_k <= records.len() {
        out.extend_from_slice(&records[start..start + window_k]);
        start += stride;
    }
    out
}
