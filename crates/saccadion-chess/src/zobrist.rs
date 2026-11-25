//! Deterministic Zobrist hashing tables for fast Transposition Table & StrategicTT lookups.

use std::sync::OnceLock;

pub struct ZobristTables {
    pub piece: [[[u64; 64]; 6]; 2],
    pub castling: [u64; 16],
    pub ep_file: [u64; 8],
    pub side: u64,
}

static ZOBRIST: OnceLock<ZobristTables> = OnceLock::new();

#[inline(always)]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn tables() -> &'static ZobristTables {
    ZOBRIST.get_or_init(|| {
        let mut seed = 0x5ACC_AD10_2026_F173u64;
        let mut piece = [[[0u64; 64]; 6]; 2];
        for c in 0..2 {
            for p in 0..6 {
                for sq in 0..64 {
                    piece[c][p][sq] = splitmix64(&mut seed);
                }
            }
        }
        let mut castling = [0u64; 16];
        for c in castling.iter_mut() {
            *c = splitmix64(&mut seed);
        }
        let mut ep_file = [0u64; 8];
        for e in ep_file.iter_mut() {
            *e = splitmix64(&mut seed);
        }
        let side = splitmix64(&mut seed);
        ZobristTables {
            piece,
            castling,
            ep_file,
            side,
        }
    })
}
