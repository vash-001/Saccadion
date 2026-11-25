//! Fundamental chess domain types with compact stack-friendly representations.

use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline(always)]
    pub const fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl Piece {
    pub const ALL: [Piece; 6] = [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ];

    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline(always)]
    pub const fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Piece::Pawn),
            1 => Some(Piece::Knight),
            2 => Some(Piece::Bishop),
            3 => Some(Piece::Rook),
            4 => Some(Piece::Queen),
            5 => Some(Piece::King),
            _ => None,
        }
    }

    /// Standard classical piece values in centipawns (used only for SEE, MVV-LVA, and LH descriptors).
    #[inline(always)]
    pub const fn cp_value(self) -> i32 {
        match self {
            Piece::Pawn => 100,
            Piece::Knight => 320,
            Piece::Bishop => 330,
            Piece::Rook => 500,
            Piece::Queen => 900,
            Piece::King => 20_000,
        }
    }

    #[inline(always)]
    pub const fn to_char(self, color: Color) -> char {
        let c = match self {
            Piece::Pawn => 'p',
            Piece::Knight => 'n',
            Piece::Bishop => 'b',
            Piece::Rook => 'r',
            Piece::Queen => 'q',
            Piece::King => 'k',
        };
        match color {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }
}

/// Board square indexed 0..63 where a1 = 0, h1 = 7, a8 = 56, h8 = 63.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Square(pub u8);

impl Square {
    #[inline(always)]
    pub const fn new(idx: u8) -> Self {
        debug_assert!(idx < 64);
        Self(idx)
    }

    #[inline(always)]
    pub const fn from_file_rank(file: u8, rank: u8) -> Self {
        debug_assert!(file < 8 && rank < 8);
        Self((rank << 3) | file)
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    #[inline(always)]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    /// Vertical rank flip (White <-> Black perspective symmetry).
    #[inline(always)]
    pub const fn flip_rank(self) -> Self {
        Self(self.0 ^ 56)
    }

    /// Horizontal file mirror (a<->h symmetry for king-bucketed feature folding).
    #[inline(always)]
    pub const fn flip_file(self) -> Self {
        Self(self.0 ^ 7)
    }

    #[inline(always)]
    pub const fn relative_to(self, color: Color) -> Self {
        match color {
            Color::White => self,
            Color::Black => self.flip_rank(),
        }
    }

    pub fn from_algebraic(s: &str) -> Option<Self> {
        let b = s.as_bytes();
        if b.len() != 2 {
            return None;
        }
        let file = b[0].wrapping_sub(b'a');
        let rank = b[1].wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Some(Self::from_file_rank(file, rank))
        } else {
            None
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        write!(f, "{}{}", file, rank)
    }
}

/// Bitmask for castling rights: WK=1, WQ=2, BK=4, BQ=8.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct CastlingRights(pub u8);

impl CastlingRights {
    pub const NONE: Self = Self(0);
    pub const WHITE_OO: u8 = 1 << 0;
    pub const WHITE_OOO: u8 = 1 << 1;
    pub const BLACK_OO: u8 = 1 << 2;
    pub const BLACK_OOO: u8 = 1 << 3;
    pub const ALL: Self = Self(0x0F);

    #[inline(always)]
    pub const fn has(self, mask: u8) -> bool {
        (self.0 & mask) != 0
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        (self.0 & 0x0F) as usize
    }
}

/// 4-bit move flag encoding compatible with classic Stockfish/ChessProgramming conventions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveFlags {
    Quiet = 0,
    DoublePawnPush = 1,
    KingCastle = 2,
    QueenCastle = 3,
    Capture = 4,
    EpCapture = 5,
    PromoKnight = 8,
    PromoBishop = 9,
    PromoRook = 10,
    PromoQueen = 11,
    PromoCaptureKnight = 12,
    PromoCaptureBishop = 13,
    PromoCaptureRook = 14,
    PromoCaptureQueen = 15,
}

/// Compact 16-bit move representation: bits 0..6 = from, bits 6..12 = to, bits 12..16 = flags.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Move(pub u16);

impl Move {
    pub const NULL: Self = Self(0);

    #[inline(always)]
    pub const fn new(from: Square, to: Square, flags: MoveFlags) -> Self {
        Self(((flags as u16) << 12) | ((to.0 as u16) << 6) | (from.0 as u16))
    }

    #[inline(always)]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn from_sq(self) -> Square {
        Square((self.0 & 0x3F) as u8)
    }

    #[inline(always)]
    pub const fn to_sq(self) -> Square {
        Square(((self.0 >> 6) & 0x3F) as u8)
    }

    #[inline(always)]
    pub const fn flags_raw(self) -> u8 {
        (self.0 >> 12) as u8
    }

    /// Packed 12-bit (from << 6 | to) policy index in 0..4096.
    #[inline(always)]
    pub const fn from_to_index(self) -> usize {
        ((self.from_sq().0 as usize) << 6) | (self.to_sq().0 as usize)
    }

    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        (self.flags_raw() & 4) != 0
    }

    #[inline(always)]
    pub const fn is_promotion(self) -> bool {
        (self.flags_raw() & 8) != 0
    }

    #[inline(always)]
    pub const fn is_en_passant(self) -> bool {
        self.flags_raw() == MoveFlags::EpCapture as u8
    }

    #[inline(always)]
    pub const fn is_castle(self) -> bool {
        let fl = self.flags_raw();
        fl == MoveFlags::KingCastle as u8 || fl == MoveFlags::QueenCastle as u8
    }

    #[inline(always)]
    pub const fn is_double_push(self) -> bool {
        self.flags_raw() == MoveFlags::DoublePawnPush as u8
    }

    #[inline(always)]
    pub const fn promotion_piece(self) -> Option<Piece> {
        if !self.is_promotion() {
            return None;
        }
        match self.flags_raw() & 3 {
            0 => Some(Piece::Knight),
            1 => Some(Piece::Bishop),
            2 => Some(Piece::Rook),
            3 => Some(Piece::Queen),
            _ => None,
        }
    }

    pub fn to_uci(self) -> String {
        if self.is_null() {
            return "0000".to_string();
        }
        let mut s = format!("{}{}", self.from_sq(), self.to_sq());
        if let Some(p) = self.promotion_piece() {
            s.push(p.to_char(Color::Black));
        }
        s
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

/// Zero-allocation stack-allocated list of moves (max legal moves in chess is 218).
#[derive(Clone)]
pub struct MoveList {
    moves: [Move; 256],
    len: usize,
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl MoveList {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            moves: [Move::NULL; 256],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < 256);
        self.moves[self.len] = mv;
        self.len += 1;
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [Move] {
        &mut self.moves[..self.len]
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }
}
