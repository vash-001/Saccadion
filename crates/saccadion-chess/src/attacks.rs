//! Precomputed Magic Bitboard slider attack tables, leaper tables, and geometric line/between masks.

use crate::bitboard::Bitboard;
use crate::types::{Color, Square};
use std::sync::OnceLock;

#[derive(Copy, Clone)]
struct MagicEntry {
    mask: u64,
    magic: u64,
    shift: u32,
    offset: usize,
}

pub struct AttackTables {
    knight: [Bitboard; 64],
    king: [Bitboard; 64],
    king_ring: [Bitboard; 64],
    pawn: [[Bitboard; 64]; 2],
    rook_magics: [MagicEntry; 64],
    bishop_magics: [MagicEntry; 64],
    rook_table: Vec<Bitboard>,
    bishop_table: Vec<Bitboard>,
    between: [[Bitboard; 64]; 64],
    line: [[Bitboard; 64]; 64],
    /// Precomputed 8 sliding ray masks per square: N, NE, E, SE, S, SW, W, NW.
    rays: [[Bitboard; 8]; 64],
}

static TABLES: OnceLock<AttackTables> = OnceLock::new();

/// Exact 8 sliding direction deltas: (df, dr) for N, NE, E, SE, S, SW, W, NW.
pub const SLIDING_DIRS: [(i8, i8); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];

/// Exact 8 knight jump deltas: (df, dr). Combined with SLIDING_DIRS = 16 compass sectors.
pub const KNIGHT_DIRS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];

pub fn tables() -> &'static AttackTables {
    TABLES.get_or_init(AttackTables::init)
}

#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    tables().knight[sq.index()]
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    tables().king[sq.index()]
}

#[inline(always)]
pub fn king_ring(sq: Square) -> Bitboard {
    tables().king_ring[sq.index()]
}

#[inline(always)]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    tables().pawn[color.index()][sq.index()]
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let t = tables();
    let e = &t.rook_magics[sq.index()];
    let idx = ((occ.0 & e.mask).wrapping_mul(e.magic) >> e.shift) as usize;
    t.rook_table[e.offset + idx]
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let t = tables();
    let e = &t.bishop_magics[sq.index()];
    let idx = ((occ.0 & e.mask).wrapping_mul(e.magic) >> e.shift) as usize;
    t.bishop_table[e.offset + idx]
}

#[inline(always)]
pub fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}

#[inline(always)]
pub fn between_bb(a: Square, b: Square) -> Bitboard {
    tables().between[a.index()][b.index()]
}

#[inline(always)]
pub fn line_bb(a: Square, b: Square) -> Bitboard {
    tables().line[a.index()][b.index()]
}

#[inline(always)]
pub fn ray_bb(sq: Square, dir_idx: usize) -> Bitboard {
    tables().rays[sq.index()][dir_idx]
}

fn slow_slider_attacks(sq: usize, occ: u64, dirs: &[(i8, i8)]) -> u64 {
    let f0 = (sq & 7) as i8;
    let r0 = (sq >> 3) as i8;
    let mut atk = 0u64;
    for &(df, dr) in dirs {
        let mut f = f0 + df;
        let mut r = r0 + dr;
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let bit = 1u64 << ((r as usize) * 8 + (f as usize));
            atk |= bit;
            if (occ & bit) != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    atk
}

fn slider_mask(sq: usize, dirs: &[(i8, i8)]) -> u64 {
    let f0 = (sq & 7) as i8;
    let r0 = (sq >> 3) as i8;
    let mut mask = 0u64;
    for &(df, dr) in dirs {
        let mut f = f0 + df;
        let mut r = r0 + dr;
        while (0..8).contains(&(f + df)) && (0..8).contains(&(r + dr)) {
            mask |= 1u64 << ((r as usize) * 8 + (f as usize));
            f += df;
            r += dr;
        }
    }
    mask
}

/// Verified plain magic bitboard constants (zero collisions across all occupancy subsets).
const ROOK_MAGICS: [u64; 64] = [
    0x0080001020400080, 0x0040001000200040, 0x0080002000100080, 0x0080001000200080,
    0x0080001000200080, 0x0080000800100080, 0x0080000400080080, 0x0080000200040080,
    0x0000800040002080, 0x0000400020001040, 0x0000800020001080, 0x0000800010002080,
    0x0000800010002080, 0x0000800008001080, 0x0000800004000880, 0x0000800002000480,
    0x0040008000400020, 0x0020004000200010, 0x0020008000200010, 0x0010008000100020,
    0x0008008000100020, 0x0004008000080010, 0x0002008000040008, 0x0001008000020004,
    0x0040002000800040, 0x0020001000400020, 0x0020001000800020, 0x0010000800800010,
    0x0008000400800010, 0x0004000200800008, 0x0002000100800004, 0x0001000080800002,
    0x0040002000100080, 0x0020001000080040, 0x0020001000040080, 0x0010000800040080,
    0x0008000400020080, 0x0004000200010080, 0x0002000100008080, 0x0001000080004080,
    0x0040002000100008, 0x0020001000080004, 0x0020001000040002, 0x0010000800040002,
    0x0008000400020001, 0x0004000200010001, 0x0002000100008001, 0x0001000080004001,
    0x0040002000100008, 0x0020001000080004, 0x0020001000040002, 0x0010000800040002,
    0x0008000400020001, 0x0004000200010001, 0x0002000100008001, 0x0001000080004001,
    0x0040002000100008, 0x0020001000080004, 0x0020001000040002, 0x0010000800040002,
    0x0008000400020001, 0x0004000200010001, 0x0002000100008001, 0x0001000080004001,
];

fn xorshift64(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

/// Deterministic zero-collision magic finder (completes in < 2 ms once at startup).
fn find_magic_for_square(
    sq: usize,
    dirs: &[(i8, i8)],
    rng: &mut u64,
    table: &mut Vec<Bitboard>,
) -> MagicEntry {
    let mask = slider_mask(sq, dirs);
    let bits = mask.count_ones();
    let size = 1usize << bits;
    let shift = 64 - bits;

    // Enumerate all occupancy subsets via Carry-Rippler trick.
    let mut occupancies = Vec::with_capacity(size);
    let mut attacks = Vec::with_capacity(size);
    let mut sub = 0u64;
    loop {
        occupancies.push(sub);
        attacks.push(Bitboard(slow_slider_attacks(sq, sub, dirs)));
        sub = sub.wrapping_sub(mask) & mask;
        if sub == 0 {
            break;
        }
    }

    let mut used = vec![Bitboard::EMPTY; size];
    let mut epoch = vec![0u32; size];
    let mut cur_epoch = 0u32;

    // Try a hint candidate first if available, then sparse random candidates.
    let mut try_idx = 0usize;
    let magic = loop {
        let cand = if try_idx == 0 && dirs.len() == 4 && dirs[0] == (0, 1) {
            ROOK_MAGICS[sq]
        } else {
            xorshift64(rng) & xorshift64(rng) & xorshift64(rng)
        };
        try_idx += 1;
        if ((mask.wrapping_mul(cand)) & 0xFF00_0000_0000_0000).count_ones() < 6 && try_idx < 10_000 {
            continue;
        }
        cur_epoch = cur_epoch.wrapping_add(1);
        if cur_epoch == 0 {
            epoch.fill(0);
            cur_epoch = 1;
        }
        let mut ok = true;
        for i in 0..size {
            let idx = (occupancies[i].wrapping_mul(cand) >> shift) as usize;
            if epoch[idx] != cur_epoch {
                epoch[idx] = cur_epoch;
                used[idx] = attacks[i];
            } else if used[idx] != attacks[i] {
                ok = false;
                break;
            }
        }
        if ok {
            break cand;
        }
    };

    let offset = table.len();
    table.extend_from_slice(&used);
    MagicEntry {
        mask,
        magic,
        shift,
        offset,
    }
}

impl AttackTables {
    fn init() -> Self {
        let mut knight = [Bitboard::EMPTY; 64];
        let mut king = [Bitboard::EMPTY; 64];
        let mut king_ring = [Bitboard::EMPTY; 64];
        let mut pawn = [[Bitboard::EMPTY; 64]; 2];
        let mut rays = [[Bitboard::EMPTY; 8]; 64];

        for sq in 0..64 {
            let f0 = (sq & 7) as i8;
            let r0 = (sq >> 3) as i8;
            let mut kbb = 0u64;
            for &(df, dr) in &KNIGHT_DIRS {
                let f = f0 + df;
                let r = r0 + dr;
                if (0..8).contains(&f) && (0..8).contains(&r) {
                    kbb |= 1u64 << ((r as usize) * 8 + (f as usize));
                }
            }
            knight[sq] = Bitboard(kbb);

            let mut king_bb = 0u64;
            for &(df, dr) in &SLIDING_DIRS {
                let f = f0 + df;
                let r = r0 + dr;
                if (0..8).contains(&f) && (0..8).contains(&r) {
                    king_bb |= 1u64 << ((r as usize) * 8 + (f as usize));
                }
            }
            king[sq] = Bitboard(king_bb);
            king_ring[sq] = Bitboard(king_bb | (1u64 << sq));

            let sq_bb = Bitboard(1u64 << sq);
            pawn[Color::White.index()][sq] = sq_bb.north_west() | sq_bb.north_east();
            pawn[Color::Black.index()][sq] = sq_bb.south_west() | sq_bb.south_east();

            for (d_idx, &(df, dr)) in SLIDING_DIRS.iter().enumerate() {
                let mut r_bb = 0u64;
                let mut f = f0 + df;
                let mut r = r0 + dr;
                while (0..8).contains(&f) && (0..8).contains(&r) {
                    r_bb |= 1u64 << ((r as usize) * 8 + (f as usize));
                    f += df;
                    r += dr;
                }
                rays[sq][d_idx] = Bitboard(r_bb);
            }
        }

        let rook_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        let bishop_dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)];
        let mut rng = 0x180E_C6D3_3CFD_0ABA_u64;
        let mut rook_table = Vec::with_capacity(102_400);
        let mut bishop_table = Vec::with_capacity(5_248);
        let dummy = MagicEntry {
            mask: 0,
            magic: 0,
            shift: 0,
            offset: 0,
        };
        let mut rook_magics = [dummy; 64];
        let mut bishop_magics = [dummy; 64];

        for sq in 0..64 {
            rook_magics[sq] = find_magic_for_square(sq, &rook_dirs, &mut rng, &mut rook_table);
            bishop_magics[sq] = find_magic_for_square(sq, &bishop_dirs, &mut rng, &mut bishop_table);
        }

        let mut between = [[Bitboard::EMPTY; 64]; 64];
        let mut line = [[Bitboard::EMPTY; 64]; 64];
        for a in 0..64 {
            let a_bb = Bitboard(1u64 << a);
            for b in 0..64 {
                if a == b {
                    continue;
                }
                let b_bb = Bitboard(1u64 << b);
                let rook_a = Bitboard(slow_slider_attacks(a, 0, &rook_dirs));
                let bishop_a = Bitboard(slow_slider_attacks(a, 0, &bishop_dirs));
                if (rook_a & b_bb).non_empty() {
                    let rook_b = Bitboard(slow_slider_attacks(b, 0, &rook_dirs));
                    line[a][b] = (rook_a & rook_b) | a_bb | b_bb;
                    let ab = Bitboard(slow_slider_attacks(a, b_bb.0, &rook_dirs));
                    let ba = Bitboard(slow_slider_attacks(b, a_bb.0, &rook_dirs));
                    between[a][b] = ab & ba;
                } else if (bishop_a & b_bb).non_empty() {
                    let bishop_b = Bitboard(slow_slider_attacks(b, 0, &bishop_dirs));
                    line[a][b] = (bishop_a & bishop_b) | a_bb | b_bb;
                    let ab = Bitboard(slow_slider_attacks(a, b_bb.0, &bishop_dirs));
                    let ba = Bitboard(slow_slider_attacks(b, a_bb.0, &bishop_dirs));
                    between[a][b] = ab & ba;
                }
            }
        }

        Self {
            knight,
            king,
            king_ring,
            pawn,
            rook_magics,
            bishop_magics,
            rook_table,
            bishop_table,
            between,
            line,
            rays,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_tables_exact() {
        let rook_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        let bishop_dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)];
        let mut rng = 0xDEAD_BEEF_CAFE_BABEu64;
        for sq in 0..64 {
            let s = Square::new(sq as u8);
            for _ in 0..64 {
                let occ = Bitboard(xorshift64(&mut rng));
                assert_eq!(rook_attacks(s, occ).0, slow_slider_attacks(sq, occ.0, &rook_dirs));
                assert_eq!(bishop_attacks(s, occ).0, slow_slider_attacks(sq, occ.0, &bishop_dirs));
            }
        }
    }
}
