//! Full chess board state with incremental Zobrist hashing, stack-copyable undo, and SEE.

use crate::attacks::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, queen_attacks, rook_attacks,
};
use crate::bitboard::Bitboard;
use crate::types::{CastlingRights, Color, Move, MoveFlags, Piece, Square};
use crate::zobrist;

pub const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const CASTLING_SPOILER: [u8; 64] = {
    let mut m = [0x0Fu8; 64];
    m[0] &= !CastlingRights::WHITE_OOO;
    m[4] &= !(CastlingRights::WHITE_OO | CastlingRights::WHITE_OOO);
    m[7] &= !CastlingRights::WHITE_OO;
    m[56] &= !CastlingRights::BLACK_OOO;
    m[60] &= !(CastlingRights::BLACK_OO | CastlingRights::BLACK_OOO);
    m[63] &= !CastlingRights::BLACK_OO;
    m
};

/// Reversible state snapshot popped on unmake_move (stack-allocated, zero heap allocation).
#[derive(Copy, Clone, Debug)]
pub struct UndoState {
    pub captured: Option<Piece>,
    pub castling: CastlingRights,
    pub ep_square: Option<Square>,
    pub halfmove_clock: u16,
    pub hash: u64,
    pub pawn_hash: u64,
}

#[derive(Clone, Debug)]
pub struct Position {
    pub pieces: [[Bitboard; 6]; 2],
    pub color_occ: [Bitboard; 2],
    pub all_occ: Bitboard,
    pub mailbox: [Option<(Color, Piece)>; 64],
    pub side_to_move: Color,
    pub castling: CastlingRights,
    pub ep_square: Option<Square>,
    pub halfmove_clock: u16,
    pub fullmove_number: u16,
    pub hash: u64,
    pub pawn_hash: u64,
    pub history: Vec<u64>,
}

impl Default for Position {
    fn default() -> Self {
        Self::startpos()
    }
}

impl Position {
    pub fn empty() -> Self {
        Self {
            pieces: [[Bitboard::EMPTY; 6]; 2],
            color_occ: [Bitboard::EMPTY; 2],
            all_occ: Bitboard::EMPTY,
            mailbox: [None; 64],
            side_to_move: Color::White,
            castling: CastlingRights::NONE,
            ep_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
            pawn_hash: 0,
            history: Vec::with_capacity(256),
        }
    }

    pub fn startpos() -> Self {
        Self::from_fen(STARTPOS_FEN).expect("valid startpos FEN")
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("Invalid FEN (too few fields): {}", fen));
        }

        let mut pos = Self::empty();
        let mut rank: i8 = 7;
        let mut file: i8 = 0;

        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    if file != 8 {
                        return Err(format!("Incomplete rank in FEN: {}", fen));
                    }
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => {
                    file += (ch as u8 - b'0') as i8;
                }
                _ => {
                    if rank < 0 || file >= 8 {
                        return Err(format!("Square out of bounds in FEN: {}", fen));
                    }
                    let color = if ch.is_ascii_uppercase() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    let piece = match ch.to_ascii_lowercase() {
                        'p' => Piece::Pawn,
                        'n' => Piece::Knight,
                        'b' => Piece::Bishop,
                        'r' => Piece::Rook,
                        'q' => Piece::Queen,
                        'k' => Piece::King,
                        _ => return Err(format!("Invalid piece char '{}' in FEN", ch)),
                    };
                    let sq = Square::from_file_rank(file as u8, rank as u8);
                    pos.put_piece(color, piece, sq);
                    file += 1;
                }
            }
        }
        if rank != 0 || file != 8 {
            return Err(format!("Incomplete board in FEN: {}", fen));
        }

        pos.side_to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            other => return Err(format!("Invalid side to move: {}", other)),
        };

        let mut rights = 0u8;
        if parts[2] != "-" {
            for ch in parts[2].chars() {
                match ch {
                    'K' => rights |= CastlingRights::WHITE_OO,
                    'Q' => rights |= CastlingRights::WHITE_OOO,
                    'k' => rights |= CastlingRights::BLACK_OO,
                    'q' => rights |= CastlingRights::BLACK_OOO,
                    _ => return Err(format!("Invalid castling char '{}'", ch)),
                }
            }
        }
        pos.castling = CastlingRights(rights);

        pos.ep_square = if parts[3] == "-" {
            None
        } else {
            Some(
                Square::from_algebraic(parts[3])
                    .ok_or_else(|| format!("Invalid ep square: {}", parts[3]))?,
            )
        };

        pos.halfmove_clock = parts
            .get(4)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);
        pos.fullmove_number = parts
            .get(5)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(1);

        pos.recompute_hashes();
        Ok(pos)
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::with_capacity(80);
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let sq = Square::from_file_rank(file, rank);
                if let Some((c, p)) = self.mailbox[sq.index()] {
                    if empty > 0 {
                        s.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    s.push(p.to_char(c));
                } else {
                    empty += 1;
                }
            }
            if empty > 0 {
                s.push((b'0' + empty) as char);
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.side_to_move == Color::White {
            'w'
        } else {
            'b'
        });
        s.push(' ');
        if self.castling.0 == 0 {
            s.push('-');
        } else {
            if self.castling.has(CastlingRights::WHITE_OO) {
                s.push('K');
            }
            if self.castling.has(CastlingRights::WHITE_OOO) {
                s.push('Q');
            }
            if self.castling.has(CastlingRights::BLACK_OO) {
                s.push('k');
            }
            if self.castling.has(CastlingRights::BLACK_OOO) {
                s.push('q');
            }
        }
        s.push(' ');
        if let Some(ep) = self.ep_square {
            s.push_str(&ep.to_string());
        } else {
            s.push('-');
        }
        s.push_str(&format!(" {} {}", self.halfmove_clock, self.fullmove_number));
        s
    }

    #[inline(always)]
    pub fn piece_bb(&self, color: Color, piece: Piece) -> Bitboard {
        self.pieces[color.index()][piece.index()]
    }

    #[inline(always)]
    pub fn king_square(&self, color: Color) -> Square {
        self.pieces[color.index()][Piece::King.index()].lsb()
    }

    #[inline(always)]
    pub fn piece_at(&self, sq: Square) -> Option<(Color, Piece)> {
        self.mailbox[sq.index()]
    }

    #[inline(always)]
    pub fn total_piece_count(&self) -> u32 {
        self.all_occ.count()
    }

    /// Game phase bucket in 0..8 based on total pieces on the board (for 8-stack evaluation).
    #[inline(always)]
    pub fn phase_bucket(&self) -> usize {
        let pc = self.all_occ.count().saturating_sub(2) as usize;
        (pc / 4).min(7)
    }

    /// Classical material count difference in centipawns from side-to-move perspective.
    pub fn material_balance_stm(&self) -> i32 {
        let mut score = 0i32;
        for &p in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
            let own = self.piece_bb(self.side_to_move, p).count() as i32;
            let opp = self.piece_bb(self.side_to_move.flip(), p).count() as i32;
            score += (own - opp) * p.cp_value();
        }
        score
    }

    #[inline(always)]
    fn put_piece(&mut self, color: Color, piece: Piece, sq: Square) {
        let bb = Bitboard::from_sq(sq);
        self.pieces[color.index()][piece.index()] |= bb;
        self.color_occ[color.index()] |= bb;
        self.all_occ |= bb;
        self.mailbox[sq.index()] = Some((color, piece));
    }

    #[inline(always)]
    fn remove_piece(&mut self, color: Color, piece: Piece, sq: Square) {
        let bb = Bitboard::from_sq(sq);
        self.pieces[color.index()][piece.index()] ^= bb;
        self.color_occ[color.index()] ^= bb;
        self.all_occ ^= bb;
        self.mailbox[sq.index()] = None;
    }

    #[inline(always)]
    fn move_piece(&mut self, color: Color, piece: Piece, from: Square, to: Square) {
        let from_to = Bitboard::from_sq(from) | Bitboard::from_sq(to);
        self.pieces[color.index()][piece.index()] ^= from_to;
        self.color_occ[color.index()] ^= from_to;
        self.all_occ ^= from_to;
        self.mailbox[from.index()] = None;
        self.mailbox[to.index()] = Some((color, piece));
    }

    pub fn recompute_hashes(&mut self) {
        let z = zobrist::tables();
        let mut h = 0u64;
        let mut ph = 0u64;
        for c in [Color::White, Color::Black] {
            for p in Piece::ALL {
                for sq in self.piece_bb(c, p) {
                    let key = z.piece[c.index()][p.index()][sq.index()];
                    h ^= key;
                    if p == Piece::Pawn {
                        ph ^= key;
                    }
                }
            }
        }
        h ^= z.castling[self.castling.index()];
        if let Some(ep) = self.ep_square {
            h ^= z.ep_file[ep.file() as usize];
        }
        if self.side_to_move == Color::Black {
            h ^= z.side;
        }
        self.hash = h;
        self.pawn_hash = ph;
    }

    #[inline(always)]
    pub fn is_square_attacked(&self, sq: Square, by_color: Color) -> bool {
        let c = by_color.index();
        if (pawn_attacks(by_color.flip(), sq) & self.pieces[c][Piece::Pawn.index()]).non_empty() {
            return true;
        }
        if (knight_attacks(sq) & self.pieces[c][Piece::Knight.index()]).non_empty() {
            return true;
        }
        if (king_attacks(sq) & self.pieces[c][Piece::King.index()]).non_empty() {
            return true;
        }
        let diag = self.pieces[c][Piece::Bishop.index()] | self.pieces[c][Piece::Queen.index()];
        if (bishop_attacks(sq, self.all_occ) & diag).non_empty() {
            return true;
        }
        let ortho = self.pieces[c][Piece::Rook.index()] | self.pieces[c][Piece::Queen.index()];
        (rook_attacks(sq, self.all_occ) & ortho).non_empty()
    }

    /// Bitboard of all pieces (both colors) attacking square `sq` given occupancy `occ`.
    #[inline(always)]
    pub fn attackers_to(&self, sq: Square, occ: Bitboard) -> Bitboard {
        let w_pawns = pawn_attacks(Color::Black, sq) & self.pieces[0][Piece::Pawn.index()];
        let b_pawns = pawn_attacks(Color::White, sq) & self.pieces[1][Piece::Pawn.index()];
        let knights = knight_attacks(sq)
            & (self.pieces[0][Piece::Knight.index()] | self.pieces[1][Piece::Knight.index()]);
        let kings = king_attacks(sq)
            & (self.pieces[0][Piece::King.index()] | self.pieces[1][Piece::King.index()]);
        let queens = self.pieces[0][Piece::Queen.index()] | self.pieces[1][Piece::Queen.index()];
        let bishops = bishop_attacks(sq, occ)
            & (self.pieces[0][Piece::Bishop.index()] | self.pieces[1][Piece::Bishop.index()] | queens);
        let rooks = rook_attacks(sq, occ)
            & (self.pieces[0][Piece::Rook.index()] | self.pieces[1][Piece::Rook.index()] | queens);
        w_pawns | b_pawns | knights | kings | bishops | rooks
    }

    /// Bitboard of all squares attacked by `color`.
    pub fn attack_map(&self, color: Color) -> Bitboard {
        let c = color.index();
        let pawns = self.pieces[c][Piece::Pawn.index()];
        let mut atk = match color {
            Color::White => pawns.north_west() | pawns.north_east(),
            Color::Black => pawns.south_west() | pawns.south_east(),
        };
        for sq in self.pieces[c][Piece::Knight.index()] {
            atk |= knight_attacks(sq);
        }
        for sq in self.pieces[c][Piece::Bishop.index()] {
            atk |= bishop_attacks(sq, self.all_occ);
        }
        for sq in self.pieces[c][Piece::Rook.index()] {
            atk |= rook_attacks(sq, self.all_occ);
        }
        for sq in self.pieces[c][Piece::Queen.index()] {
            atk |= queen_attacks(sq, self.all_occ);
        }
        if self.pieces[c][Piece::King.index()].non_empty() {
            atk |= king_attacks(self.king_square(color));
        }
        atk
    }

    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.is_square_attacked(self.king_square(self.side_to_move), self.side_to_move.flip())
    }

    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        let ksq = self.king_square(self.side_to_move);
        self.attackers_to(ksq, self.all_occ) & self.color_occ[self.side_to_move.flip().index()]
    }

    /// Bitboard of pieces belonging to `color` that are absolutely pinned to their king.
    pub fn pinned_pieces(&self, color: Color) -> Bitboard {
        let ksq = self.king_square(color);
        let us = self.color_occ[color.index()];
        let opp = color.flip().index();
        let mut pinned = Bitboard::EMPTY;

        let opp_ortho = self.pieces[opp][Piece::Rook.index()] | self.pieces[opp][Piece::Queen.index()];
        let opp_diag = self.pieces[opp][Piece::Bishop.index()] | self.pieces[opp][Piece::Queen.index()];

        let snipers = (rook_attacks(ksq, Bitboard::EMPTY) & opp_ortho)
            | (bishop_attacks(ksq, Bitboard::EMPTY) & opp_diag);
        for sniper in snipers {
            let between = crate::attacks::between_bb(ksq, sniper) & self.all_occ;
            if between.count() == 1 && (between & us).non_empty() {
                pinned |= between;
            }
        }
        pinned
    }

    /// Make a move (does not check legality; caller uses legal movegen or verifies king safety).
    pub fn make_move(&mut self, mv: Move) -> UndoState {
        let z = zobrist::tables();
        let us = self.side_to_move;
        let them = us.flip();
        let from = mv.from_sq();
        let to = mv.to_sq();
        let (_, moving_piece) = self.mailbox[from.index()].expect("no piece on from square");

        let captured = if mv.is_en_passant() {
            Some(Piece::Pawn)
        } else {
            self.mailbox[to.index()].map(|(_, p)| p)
        };

        let undo = UndoState {
            captured,
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            hash: self.hash,
            pawn_hash: self.pawn_hash,
        };

        self.history.push(self.hash);

        // Clear old ep and castling from hash
        if let Some(ep) = self.ep_square {
            self.hash ^= z.ep_file[ep.file() as usize];
        }
        self.hash ^= z.castling[self.castling.index()];
        self.ep_square = None;

        // Handle capture
        if mv.is_en_passant() {
            let cap_sq = match us {
                Color::White => Square::new(to.0 - 8),
                Color::Black => Square::new(to.0 + 8),
            };
            self.remove_piece(them, Piece::Pawn, cap_sq);
            let k = z.piece[them.index()][Piece::Pawn.index()][cap_sq.index()];
            self.hash ^= k;
            self.pawn_hash ^= k;
        } else if let Some(cap) = captured {
            self.remove_piece(them, cap, to);
            let k = z.piece[them.index()][cap.index()][to.index()];
            self.hash ^= k;
            if cap == Piece::Pawn {
                self.pawn_hash ^= k;
            }
        }

        // Move or promote piece
        if let Some(promo) = mv.promotion_piece() {
            self.remove_piece(us, Piece::Pawn, from);
            self.put_piece(us, promo, to);
            let fk = z.piece[us.index()][Piece::Pawn.index()][from.index()];
            let tk = z.piece[us.index()][promo.index()][to.index()];
            self.hash ^= fk ^ tk;
            self.pawn_hash ^= fk;
        } else {
            self.move_piece(us, moving_piece, from, to);
            let fk = z.piece[us.index()][moving_piece.index()][from.index()];
            let tk = z.piece[us.index()][moving_piece.index()][to.index()];
            self.hash ^= fk ^ tk;
            if moving_piece == Piece::Pawn {
                self.pawn_hash ^= fk ^ tk;
            }
        }

        // Handle castling rook movement
        if mv.is_castle() {
            let (r_from, r_to) = match (us, mv.flags_raw()) {
                (Color::White, f) if f == MoveFlags::KingCastle as u8 => (Square::new(7), Square::new(5)),
                (Color::White, _) => (Square::new(0), Square::new(3)),
                (Color::Black, f) if f == MoveFlags::KingCastle as u8 => (Square::new(63), Square::new(61)),
                (Color::Black, _) => (Square::new(56), Square::new(59)),
            };
            self.move_piece(us, Piece::Rook, r_from, r_to);
            self.hash ^= z.piece[us.index()][Piece::Rook.index()][r_from.index()]
                ^ z.piece[us.index()][Piece::Rook.index()][r_to.index()];
        } else if mv.is_double_push() {
            let ep_sq = Square::new((from.0 + to.0) / 2);
            self.ep_square = Some(ep_sq);
            self.hash ^= z.ep_file[ep_sq.file() as usize];
        }

        // Update castling rights
        self.castling.0 &= CASTLING_SPOILER[from.index()] & CASTLING_SPOILER[to.index()];
        self.hash ^= z.castling[self.castling.index()];

        // Clocks & side to move
        if moving_piece == Piece::Pawn || captured.is_some() {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }
        if us == Color::Black {
            self.fullmove_number += 1;
        }
        self.side_to_move = them;
        self.hash ^= z.side;

        undo
    }

    pub fn unmake_move(&mut self, mv: Move, undo: UndoState) {
        let them = self.side_to_move;
        let us = them.flip();
        self.side_to_move = us;
        if us == Color::Black {
            self.fullmove_number -= 1;
        }

        let from = mv.from_sq();
        let to = mv.to_sq();

        if mv.is_castle() {
            let (r_from, r_to) = match (us, mv.flags_raw()) {
                (Color::White, f) if f == MoveFlags::KingCastle as u8 => (Square::new(7), Square::new(5)),
                (Color::White, _) => (Square::new(0), Square::new(3)),
                (Color::Black, f) if f == MoveFlags::KingCastle as u8 => (Square::new(63), Square::new(61)),
                (Color::Black, _) => (Square::new(56), Square::new(59)),
            };
            self.move_piece(us, Piece::Rook, r_to, r_from);
        }

        if let Some(promo) = mv.promotion_piece() {
            self.remove_piece(us, promo, to);
            self.put_piece(us, Piece::Pawn, from);
        } else {
            let (_, piece) = self.mailbox[to.index()].expect("piece missing on unmake");
            self.move_piece(us, piece, to, from);
        }

        if mv.is_en_passant() {
            let cap_sq = match us {
                Color::White => Square::new(to.0 - 8),
                Color::Black => Square::new(to.0 + 8),
            };
            self.put_piece(them, Piece::Pawn, cap_sq);
        } else if let Some(cap) = undo.captured {
            self.put_piece(them, cap, to);
        }

        self.castling = undo.castling;
        self.ep_square = undo.ep_square;
        self.halfmove_clock = undo.halfmove_clock;
        self.hash = undo.hash;
        self.pawn_hash = undo.pawn_hash;
        self.history.pop();
    }

    pub fn make_null_move(&mut self) -> UndoState {
        let z = zobrist::tables();
        let undo = UndoState {
            captured: None,
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            hash: self.hash,
            pawn_hash: self.pawn_hash,
        };
        self.history.push(self.hash);
        if let Some(ep) = self.ep_square {
            self.hash ^= z.ep_file[ep.file() as usize];
            self.ep_square = None;
        }
        self.side_to_move = self.side_to_move.flip();
        self.hash ^= z.side;
        self.halfmove_clock += 1;
        undo
    }

    pub fn unmake_null_move(&mut self, undo: UndoState) {
        self.side_to_move = self.side_to_move.flip();
        self.ep_square = undo.ep_square;
        self.halfmove_clock = undo.halfmove_clock;
        self.hash = undo.hash;
        self.pawn_hash = undo.pawn_hash;
        self.history.pop();
    }

    pub fn is_repetition(&self) -> bool {
        let len = self.history.len();
        let back = (self.halfmove_clock as usize).min(len);
        let start = len - back;
        let mut i = len.saturating_sub(2);
        while i >= start {
            if self.history[i] == self.hash {
                return true;
            }
            if i < 2 {
                break;
            }
            i -= 2;
        }
        false
    }

    pub fn is_insufficient_material(&self) -> bool {
        let pawns_majors = self.pieces[0][Piece::Pawn.index()]
            | self.pieces[1][Piece::Pawn.index()]
            | self.pieces[0][Piece::Rook.index()]
            | self.pieces[1][Piece::Rook.index()]
            | self.pieces[0][Piece::Queen.index()]
            | self.pieces[1][Piece::Queen.index()];
        if pawns_majors.non_empty() {
            return false;
        }
        let minors = self.all_occ.count() - 2;
        if minors <= 1 {
            return true;
        }
        if minors == 2 {
            let wb = self.pieces[0][Piece::Bishop.index()];
            let bb = self.pieces[1][Piece::Bishop.index()];
            if wb.count() == 1 && bb.count() == 1 {
                let w_light = (wb & Bitboard::LIGHT_SQUARES).non_empty();
                let b_light = (bb & Bitboard::LIGHT_SQUARES).non_empty();
                return w_light == b_light;
            }
        }
        false
    }

    /// Static Exchange Evaluation (SEE): returns true if capture `mv` achieves >= `threshold` cp.
    pub fn see_ge(&self, mv: Move, threshold: i32) -> bool {
        let from = mv.from_sq();
        let to = mv.to_sq();
        let (_, attacker) = match self.mailbox[from.index()] {
            Some(p) => p,
            None => return false,
        };
        let victim_val = if mv.is_en_passant() {
            Piece::Pawn.cp_value()
        } else {
            self.mailbox[to.index()].map(|(_, p)| p.cp_value()).unwrap_or(0)
        };
        let mut next_victim = mv.promotion_piece().unwrap_or(attacker);
        let mut balance = victim_val + mv.promotion_piece().map(|p| p.cp_value() - Piece::Pawn.cp_value()).unwrap_or(0) - threshold;
        if balance < 0 {
            return false;
        }
        balance -= next_victim.cp_value();
        if balance >= 0 {
            return true;
        }

        let mut occ = self.all_occ ^ Bitboard::from_sq(from) ^ Bitboard::from_sq(to);
        if mv.is_en_passant() {
            let cap_sq = match self.side_to_move {
                Color::White => Square::new(to.0 - 8),
                Color::Black => Square::new(to.0 + 8),
            };
            occ ^= Bitboard::from_sq(cap_sq);
        }

        let mut attackers = self.attackers_to(to, occ) & occ;
        let diag_sliders = self.pieces[0][Piece::Bishop.index()]
            | self.pieces[1][Piece::Bishop.index()]
            | self.pieces[0][Piece::Queen.index()]
            | self.pieces[1][Piece::Queen.index()];
        let ortho_sliders = self.pieces[0][Piece::Rook.index()]
            | self.pieces[1][Piece::Rook.index()]
            | self.pieces[0][Piece::Queen.index()]
            | self.pieces[1][Piece::Queen.index()];

        let mut stm = self.side_to_move.flip();
        loop {
            let my_attackers = attackers & self.color_occ[stm.index()];
            if my_attackers.is_empty() {
                break;
            }
            // Pick least valuable attacker
            let mut lva_sq = Square::new(0);
            let mut lva_piece = Piece::King;
            for p in Piece::ALL {
                let bb = my_attackers & self.pieces[stm.index()][p.index()];
                if bb.non_empty() {
                    lva_sq = bb.lsb();
                    lva_piece = p;
                    break;
                }
            }
            occ ^= Bitboard::from_sq(lva_sq);
            if matches!(lva_piece, Piece::Pawn | Piece::Bishop | Piece::Queen) {
                attackers |= bishop_attacks(to, occ) & diag_sliders;
            }
            if matches!(lva_piece, Piece::Rook | Piece::Queen) {
                attackers |= rook_attacks(to, occ) & ortho_sliders;
            }
            attackers &= occ;

            stm = stm.flip();
            balance = -balance - 1 - lva_piece.cp_value();
            if balance >= 0 {
                if lva_piece == Piece::King && (attackers & self.color_occ[stm.index()]).non_empty() {
                    stm = stm.flip();
                }
                break;
            }
            next_victim = lva_piece;
            let _ = next_victim;
        }

        stm != self.side_to_move
    }
}
