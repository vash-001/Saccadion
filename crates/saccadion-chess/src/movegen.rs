//! Fast legal move generator and GNG tactical quiescence generator.

use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, queen_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::position::Position;
use crate::types::{CastlingRights, Color, Move, MoveFlags, MoveList, Piece, Square};

fn push_promotions(list: &mut MoveList, from: Square, to: Square, is_capture: bool) {
    if is_capture {
        list.push(Move::new(from, to, MoveFlags::PromoCaptureQueen));
        list.push(Move::new(from, to, MoveFlags::PromoCaptureRook));
        list.push(Move::new(from, to, MoveFlags::PromoCaptureBishop));
        list.push(Move::new(from, to, MoveFlags::PromoCaptureKnight));
    } else {
        list.push(Move::new(from, to, MoveFlags::PromoQueen));
        list.push(Move::new(from, to, MoveFlags::PromoRook));
        list.push(Move::new(from, to, MoveFlags::PromoBishop));
        list.push(Move::new(from, to, MoveFlags::PromoKnight));
    }
}

/// Generates all pseudo-legal moves (or tactical captures/promotions only when `tactical_only` is true).
pub fn generate_pseudo_legal(pos: &Position, list: &mut MoveList, tactical_only: bool) {
    list.clear();
    let us = pos.side_to_move;
    let them = us.flip();
    let own_occ = pos.color_occ[us.index()];
    let opp_occ = pos.color_occ[them.index()];
    let empty = !pos.all_occ;

    // 1. Pawns
    let pawns = pos.piece_bb(us, Piece::Pawn);
    let promo_rank = match us {
        Color::White => Bitboard::RANK_7,
        Color::Black => Bitboard::RANK_2,
    };
    let promo_pawns = pawns & promo_rank;
    let non_promo_pawns = pawns & !promo_rank;

    // Quiet single and double pushes
    if !tactical_only {
        let single_push = match us {
            Color::White => non_promo_pawns.north() & empty,
            Color::Black => non_promo_pawns.south() & empty,
        };
        for to in single_push {
            let from = match us {
                Color::White => Square::new(to.0 - 8),
                Color::Black => Square::new(to.0 + 8),
            };
            list.push(Move::new(from, to, MoveFlags::Quiet));
        }

        let double_push = match us {
            Color::White => (single_push & Bitboard::RANK_3).north() & empty,
            Color::Black => (single_push & Bitboard::RANK_6).south() & empty,
        };
        for to in double_push {
            let from = match us {
                Color::White => Square::new(to.0 - 16),
                Color::Black => Square::new(to.0 + 16),
            };
            list.push(Move::new(from, to, MoveFlags::DoublePawnPush));
        }
    }

    // Quiet promotions (included even in tactical_only for quiescence)
    let promo_push = match us {
        Color::White => promo_pawns.north() & empty,
        Color::Black => promo_pawns.south() & empty,
    };
    for to in promo_push {
        let from = match us {
            Color::White => Square::new(to.0 - 8),
            Color::Black => Square::new(to.0 + 8),
        };
        push_promotions(list, from, to, false);
    }

    // Pawn captures (normal + promotion captures)
    let (cap_left, cap_right, left_delta, right_delta): (Bitboard, Bitboard, i8, i8) = match us {
        Color::White => (
            non_promo_pawns.north_west() & opp_occ,
            non_promo_pawns.north_east() & opp_occ,
            -7,
            -9,
        ),
        Color::Black => (
            non_promo_pawns.south_west() & opp_occ,
            non_promo_pawns.south_east() & opp_occ,
            9,
            7,
        ),
    };
    for to in cap_left {
        let from = Square::new((to.0 as i8 + left_delta) as u8);
        list.push(Move::new(from, to, MoveFlags::Capture));
    }
    for to in cap_right {
        let from = Square::new((to.0 as i8 + right_delta) as u8);
        list.push(Move::new(from, to, MoveFlags::Capture));
    }

    let (pcap_left, pcap_right): (Bitboard, Bitboard) = match us {
        Color::White => (
            promo_pawns.north_west() & opp_occ,
            promo_pawns.north_east() & opp_occ,
        ),
        Color::Black => (
            promo_pawns.south_west() & opp_occ,
            promo_pawns.south_east() & opp_occ,
        ),
    };
    for to in pcap_left {
        let from = Square::new((to.0 as i8 + left_delta) as u8);
        push_promotions(list, from, to, true);
    }
    for to in pcap_right {
        let from = Square::new((to.0 as i8 + right_delta) as u8);
        push_promotions(list, from, to, true);
    }

    // En passant captures
    if let Some(ep_sq) = pos.ep_square {
        let ep_attackers = pawn_attacks(them, ep_sq) & pawns;
        for from in ep_attackers {
            list.push(Move::new(from, ep_sq, MoveFlags::EpCapture));
        }
    }

    let target_mask = if tactical_only { opp_occ } else { !own_occ };

    // 2. Knights
    for from in pos.piece_bb(us, Piece::Knight) {
        let attacks = knight_attacks(from) & target_mask;
        for to in attacks {
            let flag = if opp_occ.contains(to) {
                MoveFlags::Capture
            } else {
                MoveFlags::Quiet
            };
            list.push(Move::new(from, to, flag));
        }
    }

    // 3. Bishops
    for from in pos.piece_bb(us, Piece::Bishop) {
        let attacks = bishop_attacks(from, pos.all_occ) & target_mask;
        for to in attacks {
            let flag = if opp_occ.contains(to) {
                MoveFlags::Capture
            } else {
                MoveFlags::Quiet
            };
            list.push(Move::new(from, to, flag));
        }
    }

    // 4. Rooks
    for from in pos.piece_bb(us, Piece::Rook) {
        let attacks = rook_attacks(from, pos.all_occ) & target_mask;
        for to in attacks {
            let flag = if opp_occ.contains(to) {
                MoveFlags::Capture
            } else {
                MoveFlags::Quiet
            };
            list.push(Move::new(from, to, flag));
        }
    }

    // 5. Queens
    for from in pos.piece_bb(us, Piece::Queen) {
        let attacks = queen_attacks(from, pos.all_occ) & target_mask;
        for to in attacks {
            let flag = if opp_occ.contains(to) {
                MoveFlags::Capture
            } else {
                MoveFlags::Quiet
            };
            list.push(Move::new(from, to, flag));
        }
    }

    // 6. King
    let ksq = pos.king_square(us);
    for to in king_attacks(ksq) & target_mask {
        let flag = if opp_occ.contains(to) {
            MoveFlags::Capture
        } else {
            MoveFlags::Quiet
        };
        list.push(Move::new(ksq, to, flag));
    }

    // 7. Castling (checks attack clearance directly)
    if !tactical_only {
        match us {
            Color::White => {
                if pos.castling.has(CastlingRights::WHITE_OO)
                    && (pos.all_occ.0 & ((1u64 << 5) | (1u64 << 6))) == 0
                    && !pos.is_square_attacked(Square::new(4), Color::Black)
                    && !pos.is_square_attacked(Square::new(5), Color::Black)
                    && !pos.is_square_attacked(Square::new(6), Color::Black)
                {
                    list.push(Move::new(Square::new(4), Square::new(6), MoveFlags::KingCastle));
                }
                if pos.castling.has(CastlingRights::WHITE_OOO)
                    && (pos.all_occ.0 & ((1u64 << 1) | (1u64 << 2) | (1u64 << 3))) == 0
                    && !pos.is_square_attacked(Square::new(4), Color::Black)
                    && !pos.is_square_attacked(Square::new(3), Color::Black)
                    && !pos.is_square_attacked(Square::new(2), Color::Black)
                {
                    list.push(Move::new(Square::new(4), Square::new(2), MoveFlags::QueenCastle));
                }
            }
            Color::Black => {
                if pos.castling.has(CastlingRights::BLACK_OO)
                    && (pos.all_occ.0 & ((1u64 << 61) | (1u64 << 62))) == 0
                    && !pos.is_square_attacked(Square::new(60), Color::White)
                    && !pos.is_square_attacked(Square::new(61), Color::White)
                    && !pos.is_square_attacked(Square::new(62), Color::White)
                {
                    list.push(Move::new(Square::new(60), Square::new(62), MoveFlags::KingCastle));
                }
                if pos.castling.has(CastlingRights::BLACK_OOO)
                    && (pos.all_occ.0 & ((1u64 << 57) | (1u64 << 58) | (1u64 << 59))) == 0
                    && !pos.is_square_attacked(Square::new(60), Color::White)
                    && !pos.is_square_attacked(Square::new(59), Color::White)
                    && !pos.is_square_attacked(Square::new(58), Color::White)
                {
                    list.push(Move::new(Square::new(60), Square::new(58), MoveFlags::QueenCastle));
                }
            }
        }
    }
}

/// Generates strictly legal moves by verifying king safety after make/unmake.
pub fn generate_legal(pos: &mut Position, list: &mut MoveList) {
    let mut pseudo = MoveList::new();
    generate_pseudo_legal(pos, &mut pseudo, false);
    list.clear();
    let us = pos.side_to_move;
    let them = us.flip();
    for &mv in pseudo.as_slice() {
        let undo = pos.make_move(mv);
        let ksq = pos.king_square(us);
        if !pos.is_square_attacked(ksq, them) {
            list.push(mv);
        }
        pos.unmake_move(mv, undo);
    }
}

/// Generates strictly legal tactical captures and promotions for GNG Quiescence.
pub fn generate_legal_tactical(pos: &mut Position, list: &mut MoveList) {
    let mut pseudo = MoveList::new();
    generate_pseudo_legal(pos, &mut pseudo, true);
    list.clear();
    let us = pos.side_to_move;
    let them = us.flip();
    for &mv in pseudo.as_slice() {
        let undo = pos.make_move(mv);
        let ksq = pos.king_square(us);
        if !pos.is_square_attacked(ksq, them) {
            list.push(mv);
        }
        pos.unmake_move(mv, undo);
    }
}

/// Parse a UCI move string (e.g. "e2e4", "e7e8q") against legal moves in `pos`.
pub fn parse_uci_move(pos: &mut Position, uci: &str) -> Option<Move> {
    let mut legal = MoveList::new();
    generate_legal(pos, &mut legal);
    let trimmed = uci.trim();
    legal.as_slice().iter().copied().find(|m| m.to_uci() == trimmed)
}
