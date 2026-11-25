//! 64-bit bitboard wrapper with fast bit-scanning and directional shift helpers.

use crate::types::Square;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Self = Self(0);
    pub const ALL: Self = Self(u64::MAX);

    pub const FILE_A: Self = Self(0x0101_0101_0101_0101);
    pub const FILE_B: Self = Self(0x0202_0202_0202_0202);
    pub const FILE_C: Self = Self(0x0404_0404_0404_0404);
    pub const FILE_D: Self = Self(0x0808_0808_0808_0808);
    pub const FILE_E: Self = Self(0x1010_1010_1010_1010);
    pub const FILE_F: Self = Self(0x2020_2020_2020_2020);
    pub const FILE_G: Self = Self(0x4040_4040_4040_4040);
    pub const FILE_H: Self = Self(0x8080_8080_8080_8080);

    pub const NOT_FILE_A: Self = Self(!Self::FILE_A.0);
    pub const NOT_FILE_H: Self = Self(!Self::FILE_H.0);
    pub const NOT_FILE_AB: Self = Self(!(Self::FILE_A.0 | Self::FILE_B.0));
    pub const NOT_FILE_GH: Self = Self(!(Self::FILE_G.0 | Self::FILE_H.0));

    pub const RANK_1: Self = Self(0x0000_0000_0000_00FF);
    pub const RANK_2: Self = Self(0x0000_0000_0000_FF00);
    pub const RANK_3: Self = Self(0x0000_0000_00FF_0000);
    pub const RANK_4: Self = Self(0x0000_0000_FF00_0000);
    pub const RANK_5: Self = Self(0x0000_00FF_0000_0000);
    pub const RANK_6: Self = Self(0x0000_FF00_0000_0000);
    pub const RANK_7: Self = Self(0x00FF_0000_0000_0000);
    pub const RANK_8: Self = Self(0xFF00_0000_0000_0000);

    pub const LIGHT_SQUARES: Self = Self(0x55AA_55AA_55AA_55AA);
    pub const DARK_SQUARES: Self = Self(0xAA55_AA55_AA55_AA55);

    pub const CENTER_4: Self = Self(0x0000_0018_1800_0000);
    pub const EXTENDED_CENTER: Self = Self(0x0000_3C3C_3C3C_0000);

    #[inline(always)]
    pub const fn from_sq(sq: Square) -> Self {
        Self(1u64 << sq.0)
    }

    #[inline(always)]
    pub const fn file_mask(file: u8) -> Self {
        Self(Self::FILE_A.0 << file)
    }

    #[inline(always)]
    pub const fn rank_mask(rank: u8) -> Self {
        Self(Self::RANK_1.0 << (rank * 8))
    }

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn non_empty(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    #[inline(always)]
    pub const fn contains(self, sq: Square) -> bool {
        (self.0 & (1u64 << sq.0)) != 0
    }

    #[inline(always)]
    pub const fn lsb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square(self.0.trailing_zeros() as u8)
    }

    #[inline(always)]
    pub fn pop_lsb(&mut self) -> Square {
        let sq = self.lsb();
        self.0 &= self.0 - 1;
        sq
    }

    #[inline(always)]
    pub const fn north(self) -> Self {
        Self(self.0 << 8)
    }

    #[inline(always)]
    pub const fn south(self) -> Self {
        Self(self.0 >> 8)
    }

    #[inline(always)]
    pub const fn east(self) -> Self {
        Self((self.0 & Self::NOT_FILE_H.0) << 1)
    }

    #[inline(always)]
    pub const fn west(self) -> Self {
        Self((self.0 & Self::NOT_FILE_A.0) >> 1)
    }

    #[inline(always)]
    pub const fn north_east(self) -> Self {
        Self((self.0 & Self::NOT_FILE_H.0) << 9)
    }

    #[inline(always)]
    pub const fn north_west(self) -> Self {
        Self((self.0 & Self::NOT_FILE_A.0) << 7)
    }

    #[inline(always)]
    pub const fn south_east(self) -> Self {
        Self((self.0 & Self::NOT_FILE_H.0) >> 7)
    }

    #[inline(always)]
    pub const fn south_west(self) -> Self {
        Self((self.0 & Self::NOT_FILE_A.0) >> 9)
    }
}

impl Iterator for Bitboard {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            None
        } else {
            Some(self.pop_lsb())
        }
    }
}

impl BitAnd for Bitboard {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for Bitboard {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOr for Bitboard {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for Bitboard {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXor for Bitboard {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for Bitboard {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Not for Bitboard {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
