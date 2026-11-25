//! Automatic Strategic Pressure & Prophylaxis quantification for Beyond-Stockfish training
//! and live Central Complex diagnostics.

use crate::attacks::{bishop_attacks, king_ring, knight_attacks, queen_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::movegen::generate_legal;
use crate::position::Position;
use crate::types::{Color, MoveList, Piece};

/// Single-position strategic state snapshot from White's canonical frame.
#[derive(Copy, Clone, Debug, Default)]
pub struct PressureSnapshot {
    pub safe_mobility: [f32; 2],
    pub legal_count: [f32; 2],
    pub king_ring_pressure: [f32; 2],
    pub space_control: [f32; 2],
    pub max_piece_activity: [f32; 2],
    pub fortress_index: f32,
}

/// 5-dimensional 10-20 ply strategic pressure target vector from STM perspective:
/// [0] mobility_collapse   : fractional drop in opponent safe mobility over +6..+12 plies
/// [1] entropy_squeeze     : reduction in opponent playable choices (Zugzwang pressure)
/// [2] king_pressure_delta : buildup of pressure on opponent's 9-square king ring
/// [3] space_outpost_gain  : territorial space advantage gained by +12 plies
/// [4] prophylaxis_score   : suppression of opponent's most active piece immediately after move
#[derive(Copy, Clone, Debug, Default)]
pub struct StrategicPressureTargets {
    pub values: [f32; 5],
}

fn count_safe_mobility(pos: &Position, color: Color) -> (f32, f32) {
    let opp = color.flip();
    let opp_pawns = pos.piece_bb(opp, Piece::Pawn);
    let opp_pawn_atk = match opp {
        Color::White => opp_pawns.north_west() | opp_pawns.north_east(),
        Color::Black => opp_pawns.south_west() | opp_pawns.south_east(),
    };
    let safe_mask = !pos.color_occ[color.index()] & !opp_pawn_atk;

    let mut total = 0.0f32;
    let mut best_piece = 0.0f32;

    for sq in pos.piece_bb(color, Piece::Knight) {
        let m = (knight_attacks(sq) & safe_mask).count() as f32;
        total += m;
        best_piece = best_piece.max(m);
    }
    for sq in pos.piece_bb(color, Piece::Bishop) {
        let m = (bishop_attacks(sq, pos.all_occ) & safe_mask).count() as f32;
        total += m;
        best_piece = best_piece.max(m);
    }
    for sq in pos.piece_bb(color, Piece::Rook) {
        let m = (rook_attacks(sq, pos.all_occ) & safe_mask).count() as f32;
        total += m;
        best_piece = best_piece.max(m);
    }
    for sq in pos.piece_bb(color, Piece::Queen) {
        let m = (queen_attacks(sq, pos.all_occ) & safe_mask).count() as f32;
        total += m * 0.8;
        best_piece = best_piece.max(m);
    }
    (total, best_piece)
}

/// Computes a fast snapshot of strategic mobility, king pressure, space, and fortress state.
pub fn compute_pressure_snapshot(pos: &Position) -> PressureSnapshot {
    let mut snap = PressureSnapshot::default();
    let atk_w = pos.attack_map(Color::White);
    let atk_b = pos.attack_map(Color::Black);

    let (mob_w, max_w) = count_safe_mobility(pos, Color::White);
    let (mob_b, max_b) = count_safe_mobility(pos, Color::Black);
    snap.safe_mobility = [mob_w, mob_b];
    snap.max_piece_activity = [max_w, max_b];

    // King ring pressure exerted BY White on Black's king (index 0) and BY Black on White's king (index 1)
    let bk_ring = king_ring(pos.king_square(Color::Black));
    let wk_ring = king_ring(pos.king_square(Color::White));
    snap.king_ring_pressure[0] = (atk_w & bk_ring).count() as f32;
    snap.king_ring_pressure[1] = (atk_b & wk_ring).count() as f32;

    // Space control in opponent half & extended center
    let black_half = Bitboard::RANK_5 | Bitboard::RANK_6 | Bitboard::RANK_7 | Bitboard::RANK_8;
    let white_half = Bitboard::RANK_1 | Bitboard::RANK_2 | Bitboard::RANK_3 | Bitboard::RANK_4;
    snap.space_control[0] =
        (atk_w & black_half).count() as f32 + (atk_w & Bitboard::EXTENDED_CENTER).count() as f32 * 0.5;
    snap.space_control[1] =
        (atk_b & white_half).count() as f32 + (atk_b & Bitboard::EXTENDED_CENTER).count() as f32 * 0.5;

    // Legal move count for current side to move (approximate for opponent via safe mobility)
    let mut tmp = pos.clone();
    let mut ml = MoveList::new();
    generate_legal(&mut tmp, &mut ml);
    let stm_i = pos.side_to_move.index();
    snap.legal_count[stm_i] = ml.len() as f32;
    snap.legal_count[1 - stm_i] = snap.safe_mobility[1 - stm_i].max(1.0);

    // Fortress heuristic index in [0, 1]
    let wp = pos.piece_bb(Color::White, Piece::Pawn);
    let bp = pos.piece_bb(Color::Black, Piece::Pawn);
    let blocked = (wp.north() & bp).count() as f32;
    let total_p = (wp.count() + bp.count()).max(1) as f32;
    let lock_ratio = (blocked * 2.0 / total_p).min(1.0);
    let wb = pos.piece_bb(Color::White, Piece::Bishop);
    let bb = pos.piece_bb(Color::Black, Piece::Bishop);
    let ocb = wb.count() == 1
        && bb.count() == 1
        && ((wb & Bitboard::LIGHT_SQUARES).non_empty() != (bb & Bitboard::LIGHT_SQUARES).non_empty());
    let no_majors = (pos.piece_bb(Color::White, Piece::Queen)
        | pos.piece_bb(Color::Black, Piece::Queen)
        | pos.piece_bb(Color::White, Piece::Rook)
        | pos.piece_bb(Color::Black, Piece::Rook))
    .is_empty();
    snap.fortress_index = if pos.is_insufficient_material() {
        1.0
    } else if ocb && no_majors {
        (0.5 + 0.5 * lock_ratio).min(1.0)
    } else {
        (lock_ratio * 0.65 * (pos.halfmove_clock as f32 / 40.0).min(1.0)).min(1.0)
    };

    snap
}

/// Computes the 5 long-horizon Strategic Pressure targets for ply `k` by peeking ahead inside
/// the full game trajectory `snapshots` (+2, +6, +12, +20 plies).
pub fn compute_trajectory_pressure(
    snapshots: &[PressureSnapshot],
    stm_colors: &[Color],
    ply: usize,
) -> StrategicPressureTargets {
    let n = snapshots.len();
    if ply >= n {
        return StrategicPressureTargets::default();
    }
    let us = stm_colors[ply].index();
    let opp = 1 - us;
    let cur = snapshots[ply];
    let p_next = snapshots[(ply + 2).min(n - 1)];
    let p6 = snapshots[(ply + 6).min(n - 1)];
    let p12 = snapshots[(ply + 12).min(n - 1)];
    let p20 = snapshots[(ply + 20).min(n - 1)];

    // 1. Opponent mobility collapse over +6..+12..+20 plies
    let future_opp_mob = 0.5 * p12.safe_mobility[opp] + 0.3 * p6.safe_mobility[opp] + 0.2 * p20.safe_mobility[opp];
    let mob_collapse = ((cur.safe_mobility[opp] - future_opp_mob) / (cur.safe_mobility[opp] + 4.0)).clamp(-1.0, 1.0);

    // 2. Legal-move entropy / Zugzwang squeeze
    let future_opp_choices = 0.6 * p12.legal_count[opp] + 0.4 * p6.legal_count[opp];
    let entropy_squeeze = ((cur.legal_count[opp] - future_opp_choices) / (cur.legal_count[opp] + 6.0)).clamp(-1.0, 1.0);

    // 3. King-zone pressure buildup on opponent's king minus counter-pressure on ours
    let future_kpress = 0.5 * p6.king_ring_pressure[us] + 0.5 * p12.king_ring_pressure[us];
    let future_kcounter = 0.5 * p6.king_ring_pressure[opp] + 0.5 * p12.king_ring_pressure[opp];
    let king_delta = (((future_kpress - cur.king_ring_pressure[us]) - 0.5 * (future_kcounter - cur.king_ring_pressure[opp])) / 6.0)
        .clamp(-1.0, 1.0);

    // 4. Net territorial space gain by +12 plies
    let cur_space_diff = cur.space_control[us] - cur.space_control[opp];
    let fut_space_diff = p12.space_control[us] - p12.space_control[opp];
    let space_gain = ((fut_space_diff - cur_space_diff) / 16.0).clamp(-1.0, 1.0);

    // 5. Prophylaxis: immediate drop in opponent's strongest active piece after our move
    let prophylaxis = ((cur.max_piece_activity[opp] - p_next.max_piece_activity[opp])
        / (cur.max_piece_activity[opp] + 3.0))
        .clamp(-1.0, 1.0);

    StrategicPressureTargets {
        values: [mob_collapse, entropy_squeeze, king_delta, space_gain, prophylaxis],
    }
}
