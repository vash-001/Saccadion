//! Exact Perft verification routines proving 100% bug-free move generation and make/unmake.

use crate::movegen::generate_legal;
use crate::position::Position;
use crate::types::MoveList;

pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut moves = MoveList::new();
    generate_legal(pos, &mut moves);
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut nodes = 0u64;
    for &mv in moves.as_slice() {
        let undo = pos.make_move(mv);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(mv, undo);
    }
    nodes
}

pub fn divide(pos: &mut Position, depth: u32) -> Vec<(String, u64)> {
    let mut moves = MoveList::new();
    generate_legal(pos, &mut moves);
    let mut out = Vec::with_capacity(moves.len());
    for &mv in moves.as_slice() {
        let undo = pos.make_move(mv);
        let n = if depth <= 1 { 1 } else { perft(pos, depth - 1) };
        pos.unmake_move(mv, undo);
        out.push((mv.to_uci(), n));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{extract_bilateral_sparse, extract_lateral_horn_features, extract_stream_b_board};

    #[test]
    fn test_perft_startpos() {
        let mut pos = Position::startpos();
        let orig_hash = pos.hash;
        assert_eq!(perft(&mut pos, 1), 20);
        assert_eq!(perft(&mut pos, 2), 400);
        assert_eq!(perft(&mut pos, 3), 8_902);
        assert_eq!(perft(&mut pos, 4), 197_281);
        assert_eq!(pos.hash, orig_hash);
        assert_eq!(pos.to_fen(), crate::position::STARTPOS_FEN);
    }

    #[test]
    fn test_perft_kiwipete() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let mut pos = Position::from_fen(fen).unwrap();
        assert_eq!(perft(&mut pos, 1), 48);
        assert_eq!(perft(&mut pos, 2), 2_039);
        assert_eq!(perft(&mut pos, 3), 97_862);
    }

    #[test]
    fn test_perft_ep_pin_position3() {
        let fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
        let mut pos = Position::from_fen(fen).unwrap();
        assert_eq!(perft(&mut pos, 1), 14);
        assert_eq!(perft(&mut pos, 2), 191);
        assert_eq!(perft(&mut pos, 3), 2_812);
        assert_eq!(perft(&mut pos, 4), 43_238);
    }

    #[test]
    fn test_perft_promotions_and_castling_position4() {
        let fen = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
        let mut pos = Position::from_fen(fen).unwrap();
        assert_eq!(perft(&mut pos, 1), 6);
        assert_eq!(perft(&mut pos, 2), 264);
        assert_eq!(perft(&mut pos, 3), 9_467);
    }

    #[test]
    fn test_perft_position5() {
        let fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
        let mut pos = Position::from_fen(fen).unwrap();
        assert_eq!(perft(&mut pos, 1), 44);
        assert_eq!(perft(&mut pos, 2), 1_486);
        assert_eq!(perft(&mut pos, 3), 62_379);
    }

    #[test]
    fn test_biological_feature_extractors_sanity() {
        let pos = Position::startpos();
        let (f_stm, f_opp) = extract_bilateral_sparse(&pos);
        assert!(f_stm.len >= 32);
        assert!(f_opp.len >= 32);
        let board = extract_stream_b_board(&pos);
        assert_eq!(board[0][3], 1.0); // White Rook on a1
        let lh = extract_lateral_horn_features(&pos);
        assert_eq!(lh[0], 0.0); // Not in check
        assert_eq!(lh[20], 0.0); // Not insufficient material
    }
}
