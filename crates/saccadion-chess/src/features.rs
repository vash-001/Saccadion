//! Biological sensory feature extractors shared between Rust inference and JAX TPU training.
//!
//! 1. Bilateral HalfKAv2_hm + FullRays sparse indices (Stream A isolated hemispheres).
//! 2. T4/T5 ON-OFF 16-direction ray + 48 Lobula Columnar square tensor [64, 48] (Stream B).
//! 3. Lateral Horn innate tactical-hazard & fortress descriptor vector [32] (98.2%-I Veto).

use crate::attacks::{
    bishop_attacks, king_ring, knight_attacks, pawn_attacks, ray_bb, rook_attacks, KNIGHT_DIRS,
    SLIDING_DIRS,
};
use crate::bitboard::Bitboard;
use crate::position::Position;
use crate::types::{Color, Piece, Square};

/// Number of king buckets with horizontal mirroring (files a-d folded into 16 spatial buckets).
pub const NUM_KING_BUCKETS: usize = 16;
/// Piece-square features per king bucket: 2 relative colors * 6 pieces * 64 squares = 768.
pub const HALFKA_PER_BUCKET: usize = 768;
/// Total HalfKAv2_hm piece-square sparse features = 12,288.
pub const HALFKA_DIM: usize = NUM_KING_BUCKETS * HALFKA_PER_BUCKET;
/// Compact FullRays ON/OFF (256) + LC4/LPLC2 Pairwise Threat-Edge Graph (1,024) = 1,280.
pub const RAY_SPARSE_DIM: usize = 256 + 1024;
/// Total sparse input dimension per isolated hemisphere for Stream A (12,288 + 1,280 = 13,568).
pub const STREAM_A_SPARSE_DIM: usize = HALFKA_DIM + RAY_SPARSE_DIM;

/// Square feature width for Stream B (Optic Lobe T4/T5 ON-OFF + Lobula Columnar features).
pub const STREAM_B_SQ_FEAT_DIM: usize = 48;
/// Innate Lateral Horn tactical & fortress descriptor dimension.
pub const LATERAL_HORN_DIM: usize = 32;

/// Maps relative king square (rank 0..7, file 0..3 after horizontal mirror) into 16 buckets.
#[inline(always)]
pub fn king_bucket(king_sq: Square, perspective: Color) -> (usize, bool) {
    let rel = king_sq.relative_to(perspective);
    let mirror = rel.file() >= 4;
    let f = if mirror { 7 - rel.file() } else { rel.file() } as usize;
    let r = rel.rank() as usize;
    // 4 files x 4 rank-pairs = 16 compact king buckets
    let bucket = (r / 2) * 4 + f;
    (bucket, mirror)
}

#[inline(always)]
pub fn halfka_feature_index(
    perspective: Color,
    king_sq: Square,
    piece_color: Color,
    piece: Piece,
    sq: Square,
) -> usize {
    let (bucket, mirror) = king_bucket(king_sq, perspective);
    let mut rel_sq = sq.relative_to(perspective);
    if mirror {
        rel_sq = rel_sq.flip_file();
    }
    let rel_color = if piece_color == perspective { 0 } else { 1 };
    bucket * HALFKA_PER_BUCKET + (rel_color * 6 + piece.index()) * 64 + rel_sq.index()
}

/// Active sparse feature index list for one hemisphere (zero heap allocation).
#[derive(Clone, Debug)]
pub struct SparseFeatureList {
    pub indices: [u16; 160],
    pub len: usize,
}

impl Default for SparseFeatureList {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseFeatureList {
    pub const fn new() -> Self {
        Self {
            indices: [0; 160],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, idx: usize) {
        if self.len < 160 {
            self.indices[self.len] = idx as u16;
            self.len += 1;
        }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u16] {
        &self.indices[..self.len]
    }
}

/// Extracts isolated sparse feature lists for (STM hemisphere, Opponent hemisphere).
pub fn extract_bilateral_sparse(pos: &Position) -> (SparseFeatureList, SparseFeatureList) {
    let stm = pos.side_to_move;
    let opp = stm.flip();
    let stm_atk = pos.attack_map(stm);
    let opp_atk = pos.attack_map(opp);
    let stm_pin = pos.pinned_pieces(stm);
    let opp_pin = pos.pinned_pieces(opp);

    let f_stm = extract_one_hemisphere(pos, stm, stm_atk, opp_atk, stm_pin);
    let f_opp = extract_one_hemisphere(pos, opp, opp_atk, stm_atk, opp_pin);
    (f_stm, f_opp)
}

fn extract_one_hemisphere(
    pos: &Position,
    persp: Color,
    own_atk: Bitboard,
    opp_atk: Bitboard,
    own_pin: Bitboard,
) -> SparseFeatureList {
    let mut list = SparseFeatureList::new();
    let ksq = pos.king_square(persp);
    let (_, mirror) = king_bucket(ksq, persp);

    // 1. HalfKAv2_hm piece features
    for c in [Color::White, Color::Black] {
        for p in Piece::ALL {
            for sq in pos.piece_bb(c, p) {
                let idx = halfka_feature_index(persp, ksq, c, p, sq);
                list.push(idx);
            }
        }
    }

    // 2. Compact T4/T5 FullRays ON/OFF sparse indicators (up to 128 active bits)
    let own_occ = pos.color_occ[persp.index()];
    let opp_occ = pos.color_occ[persp.flip().index()];
    let contested_own = own_occ & opp_atk;
    let contested_opp = opp_occ & own_atk;
    let weak_undefended = (own_occ & opp_atk) & !own_atk;

    for sq in contested_own {
        let mut rsq = sq.relative_to(persp);
        if mirror {
            rsq = rsq.flip_file();
        }
        list.push(HALFKA_DIM + rsq.index());
    }
    for sq in contested_opp {
        let mut rsq = sq.relative_to(persp);
        if mirror {
            rsq = rsq.flip_file();
        }
        list.push(HALFKA_DIM + 64 + rsq.index());
    }
    for sq in own_pin {
        let mut rsq = sq.relative_to(persp);
        if mirror {
            rsq = rsq.flip_file();
        }
        list.push(HALFKA_DIM + 128 + rsq.index());
    }
    for sq in weak_undefended {
        let mut rsq = sq.relative_to(persp);
        if mirror {
            rsq = rsq.flip_file();
        }
        list.push(HALFKA_DIM + 192 + rsq.index());
    }

    // 3. LC4/LPLC2 Optic Glomeruli Pairwise Threat-Edge Graph (attacker_p x victim_p x quadrant x edge_cat)
    //    Captures direct piece-on-piece attack & defense interactions in 1024 sparse buckets!
    let all_occ = pos.all_occ;
    let mut seen_lc4 = [0u64; 16]; // 1024-bit deduplication mask
    for c in [Color::White, Color::Black] {
        let is_own_atk = c == persp;
        for p in Piece::ALL {
            for u in pos.piece_bb(c, p) {
                let atks = piece_attacks_from(pos, c, p, u, all_occ) & all_occ;
                for v in atks {
                    if let Some((vc, vp)) = pos.piece_at(v) {
                        let mut rv = v.relative_to(persp);
                        if mirror {
                            rv = rv.flip_file();
                        }
                        let quad = (((rv.rank() as usize) >> 2) << 1) | ((rv.file() as usize) >> 2);
                        let is_attack = vc != c;
                        let edge_cat = (if is_own_atk { 0 } else { 2 }) + (if is_attack { 0 } else { 1 });
                        let pair_id = (p.index() * 6 + vp.index()) * 4 + quad;
                        let lc4 = (edge_cat * 256 + (pair_id & 255)) & 1023;
                        let word = lc4 >> 6;
                        let bit = 1u64 << (lc4 & 63);
                        if (seen_lc4[word] & bit) == 0 {
                            seen_lc4[word] |= bit;
                            list.push(HALFKA_DIM + 256 + lc4);
                        }
                    }
                }
            }
        }
    }

    list
}

#[inline(always)]
fn piece_attacks_from(pos: &Position, c: Color, p: Piece, sq: Square, occ: Bitboard) -> Bitboard {
    let _ = pos;
    match p {
        Piece::Pawn => pawn_attacks(c, sq),
        Piece::Knight => knight_attacks(sq),
        Piece::Bishop => bishop_attacks(sq, occ),
        Piece::Rook => rook_attacks(sq, occ),
        Piece::Queen => bishop_attacks(sq, occ) | rook_attacks(sq, occ),
        Piece::King => king_ring(sq) & !Bitboard::from_sq(sq),
    }
}

/// Extracts the exact 6-channel FlyWire Bi-Directional X-Ray Chess-Graph Adjacency Matrix (STM-oriented):
/// - Channel 0: STM Attack Edge (STM piece u -> empty or enemy square v)                [+ACh Excitatory]
/// - Channel 1: STM Defend Edge (STM piece u -> own STM piece v)                        [+ACh Excitatory]
/// - Channel 2: Opp Attack Edge (Opp piece u -> empty or STM square v)                  [-GABA Inhibitory]
/// - Channel 3: Opp Defend Edge (Opp piece u -> own Opp piece v)                        [-GABA Inhibitory]
/// - Channel 4: STM X-Ray Through-Blocker (STM slider u --[1st blocker]--> square v)    [+ACh X-Ray]
/// - Channel 5: Opp X-Ray Through-Blocker (Opp slider u --[1st blocker]--> square v)    [-GABA X-Ray]
/// Each entry `adj[k][rel_u]` is a 64-bit bitmask of destination squares `rel_v` in STM orientation.
pub fn extract_chess_graph_adj(pos: &Position) -> [[u64; 64]; 6] {
    let mut adj = [[0u64; 64]; 6];
    let stm = pos.side_to_move;
    let opp = stm.flip();
    let all_occ = pos.all_occ;
    let stm_occ = pos.color_occ[stm.index()];
    let opp_occ = pos.color_occ[opp.index()];

    let orient_bb = |bb: Bitboard| -> u64 {
        if stm == Color::White {
            bb.0
        } else {
            bb.0.swap_bytes()
        }
    };

    for p in Piece::ALL {
        let is_slider = matches!(p, Piece::Bishop | Piece::Rook | Piece::Queen);
        for u in pos.piece_bb(stm, p) {
            let atks = piece_attacks_from(pos, stm, p, u, all_occ);
            let rel_u = u.relative_to(stm).index();
            adj[0][rel_u] = orient_bb(atks & !stm_occ);
            adj[1][rel_u] = orient_bb(atks & stm_occ);
            if is_slider {
                let occ_no_block = all_occ & !atks;
                let xray = piece_attacks_from(pos, stm, p, u, occ_no_block) & !atks;
                adj[4][rel_u] = orient_bb(xray);
            }
        }
        for u in pos.piece_bb(opp, p) {
            let atks = piece_attacks_from(pos, opp, p, u, all_occ);
            let rel_u = u.relative_to(stm).index();
            adj[2][rel_u] = orient_bb(atks & !opp_occ);
            adj[3][rel_u] = orient_bb(atks & opp_occ);
            if is_slider {
                let occ_no_block = all_occ & !atks;
                let xray = piece_attacks_from(pos, opp, p, u, occ_no_block) & !atks;
                adj[5][rel_u] = orient_bb(xray);
            }
        }
    }
    adj
}

/// Extracts Stream B's 64-square biological Optic Lobe tensor [64][48] from STM perspective.
/// Layout per square (48 floats):
/// - 0..12  : One-hot piece identity (6 STM pieces + 6 Opp pieces)
/// - 12..20 : T4 ON 8-ray incoming own pressure (normalized piece weight along N..NW)
/// - 20..28 : T5 OFF/Inhibitory 8-ray incoming opponent pressure / blocking along N..NW
/// - 28..36 : 8-direction knight pressure (ON own minus OFF opponent)
/// - 36..48 : 12 Lobula Columnar local descriptors (STM atk, Opp atk, pin, king-ring, passed pawn,
///            open file, file/rank coords, castling, halfmove urgency, square color, occupancy)
pub fn extract_stream_b_board(pos: &Position) -> [[f32; STREAM_B_SQ_FEAT_DIM]; 64] {
    let mut out = [[0.0f32; STREAM_B_SQ_FEAT_DIM]; 64];
    let stm = pos.side_to_move;
    let opp = stm.flip();
    let stm_atk = pos.attack_map(stm);
    let opp_atk = pos.attack_map(opp);
    let stm_pin = pos.pinned_pieces(stm);
    let opp_pin = pos.pinned_pieces(opp);
    let stm_kring = king_ring(pos.king_square(stm));
    let opp_kring = king_ring(pos.king_square(opp));
    let all_pawns = pos.piece_bb(Color::White, Piece::Pawn) | pos.piece_bb(Color::Black, Piece::Pawn);
    let halfmove_norm = (pos.halfmove_clock as f32 / 100.0).min(1.0);
    let castling_own = match stm {
        Color::White => (pos.castling.0 & 0x03).count_ones() as f32 * 0.5,
        Color::Black => (pos.castling.0 & 0x0C).count_ones() as f32 * 0.5,
    };

    for rel_idx in 0..64 {
        let rel_sq = Square::new(rel_idx as u8);
        let abs_sq = rel_sq.relative_to(stm);
        let row = &mut out[rel_idx];

        // 0..12: Piece identity
        if let Some((c, p)) = pos.piece_at(abs_sq) {
            let offset = if c == stm { 0 } else { 6 };
            row[offset + p.index()] = 1.0;
            row[47] = 1.0; // occupied flag
        }

        // 12..28: 8-ray T4 (ON own) & T5 (OFF/inhibitory opp) first-blocker ray pressure
        for (dir_idx, &(df, dr)) in SLIDING_DIRS.iter().enumerate() {
            // Flip vertical direction if black to move so compass is always STM-oriented
            let eff_dr = if stm == Color::White { dr } else { -dr };
            let mut f = abs_sq.file() as i8 + df;
            let mut r = abs_sq.rank() as i8 + eff_dr;
            let mut dist = 1.0f32;
            while (0..8).contains(&f) && (0..8).contains(&r) {
                let scan_sq = Square::from_file_rank(f as u8, r as u8);
                if let Some((c, p)) = pos.piece_at(scan_sq) {
                    let is_diag = (dir_idx & 1) == 1;
                    let slider_matches = match p {
                        Piece::Queen => true,
                        Piece::Bishop => is_diag,
                        Piece::Rook => !is_diag,
                        Piece::King => dist == 1.0,
                        Piece::Pawn => dist == 1.0 && is_diag,
                        Piece::Knight => false,
                    };
                    let weight = if slider_matches {
                        1.0 / (0.5 + 0.5 * dist)
                    } else {
                        0.25 / dist
                    };
                    if c == stm {
                        row[12 + dir_idx] = weight;
                    } else {
                        row[20 + dir_idx] = weight;
                    }
                    break;
                }
                dist += 1.0;
                f += df;
                r += eff_dr;
            }
        }

        // 28..36: 8 knight compass vectors
        for (k_idx, &(df, dr)) in KNIGHT_DIRS.iter().enumerate() {
            let eff_dr = if stm == Color::White { dr } else { -dr };
            let f = abs_sq.file() as i8 + df;
            let r = abs_sq.rank() as i8 + eff_dr;
            if (0..8).contains(&f) && (0..8).contains(&r) {
                let scan_sq = Square::from_file_rank(f as u8, r as u8);
                if let Some((c, Piece::Knight)) = pos.piece_at(scan_sq) {
                    row[28 + k_idx] = if c == stm { 1.0 } else { -1.0 };
                }
            }
        }

        // 36..47: Lobula Columnar local descriptors
        row[36] = if stm_atk.contains(abs_sq) { 1.0 } else { 0.0 };
        row[37] = if opp_atk.contains(abs_sq) { 1.0 } else { 0.0 };
        row[38] = if stm_pin.contains(abs_sq) {
            1.0
        } else if opp_pin.contains(abs_sq) {
            -1.0
        } else {
            0.0
        };
        row[39] = if stm_kring.contains(abs_sq) { 1.0 } else { 0.0 };
        row[40] = if opp_kring.contains(abs_sq) { 1.0 } else { 0.0 };
        let file_bb = Bitboard::file_mask(abs_sq.file());
        row[41] = if (all_pawns & file_bb).is_empty() { 1.0 } else { 0.0 };
        row[42] = if is_passed_pawn_sq(pos, abs_sq) { 1.0 } else { 0.0 };
        row[43] = rel_sq.file() as f32 / 7.0;
        row[44] = rel_sq.rank() as f32 / 7.0;
        row[45] = castling_own;
        row[46] = halfmove_norm;
    }

    out
}

fn is_passed_pawn_sq(pos: &Position, sq: Square) -> bool {
    let (color, piece) = match pos.piece_at(sq) {
        Some(cp) => cp,
        None => return false,
    };
    if piece != Piece::Pawn {
        return false;
    }
    let opp_pawns = pos.piece_bb(color.flip(), Piece::Pawn);
    let f = sq.file();
    let r = sq.rank();
    let mut files_mask = Bitboard::file_mask(f);
    if f > 0 {
        files_mask |= Bitboard::file_mask(f - 1);
    }
    if f < 7 {
        files_mask |= Bitboard::file_mask(f + 1);
    }
    let mut ahead = Bitboard::EMPTY;
    match color {
        Color::White => {
            for rr in (r + 1)..8 {
                ahead |= Bitboard::rank_mask(rr);
            }
        }
        Color::Black => {
            for rr in 0..r {
                ahead |= Bitboard::rank_mask(rr);
            }
        }
    }
    (opp_pawns & files_mask & ahead).is_empty()
}

/// Extracts the 32-dim innate Lateral Horn descriptor vector for fast tactical-hazard & fortress veto.
pub fn extract_lateral_horn_features(pos: &Position) -> [f32; LATERAL_HORN_DIM] {
    let mut v = [0.0f32; LATERAL_HORN_DIM];
    let stm = pos.side_to_move;
    let opp = stm.flip();
    let stm_atk = pos.attack_map(stm);
    let opp_atk = pos.attack_map(opp);
    let stm_ksq = pos.king_square(stm);
    let opp_ksq = pos.king_square(opp);
    let stm_ring = king_ring(stm_ksq);
    let opp_ring = king_ring(opp_ksq);

    // 0..4: Check & Pin danger
    let checkers = pos.checkers();
    v[0] = (checkers.count() as f32).min(2.0) * 0.5;
    v[1] = if checkers.count() >= 2 { 1.0 } else { 0.0 };
    v[2] = (pos.pinned_pieces(stm).count() as f32 / 4.0).min(1.0);
    v[3] = (pos.pinned_pieces(opp).count() as f32 / 4.0).min(1.0);

    // 4..8: King-ring exposure & safety
    v[4] = (stm_ring & opp_atk).count() as f32 / 9.0;
    v[5] = (opp_ring & stm_atk).count() as f32 / 9.0;
    v[6] = (stm_ring & pos.piece_bb(stm, Piece::Pawn)).count() as f32 / 4.0;
    v[7] = (opp_ring & pos.piece_bb(opp, Piece::Pawn)).count() as f32 / 4.0;

    // 8..12: Hanging material & attacked high-value pieces
    let mut stm_hanging_val = 0i32;
    let mut opp_hanging_val = 0i32;
    let mut stm_en_prise_cnt = 0f32;
    let mut opp_en_prise_cnt = 0f32;
    for &p in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        for sq in pos.piece_bb(stm, p) {
            if opp_atk.contains(sq) && !stm_atk.contains(sq) {
                stm_hanging_val = stm_hanging_val.max(p.cp_value());
                stm_en_prise_cnt += 1.0;
            }
        }
        for sq in pos.piece_bb(opp, p) {
            if stm_atk.contains(sq) && !opp_atk.contains(sq) {
                opp_hanging_val = opp_hanging_val.max(p.cp_value());
                opp_en_prise_cnt += 1.0;
            }
        }
    }
    v[8] = (stm_hanging_val as f32 / 900.0).min(1.0);
    v[9] = (opp_hanging_val as f32 / 900.0).min(1.0);
    v[10] = (stm_en_prise_cnt / 4.0).min(1.0);
    v[11] = (opp_en_prise_cnt / 4.0).min(1.0);

    // 12..16: Pawn promotion race urgency
    let mut stm_best_passer = 0.0f32;
    let mut opp_best_passer = 0.0f32;
    for sq in pos.piece_bb(stm, Piece::Pawn) {
        if is_passed_pawn_sq(pos, sq) {
            let r = sq.relative_to(stm).rank() as f32;
            stm_best_passer = stm_best_passer.max((r / 6.0).min(1.0));
        }
    }
    for sq in pos.piece_bb(opp, Piece::Pawn) {
        if is_passed_pawn_sq(pos, sq) {
            let r = sq.relative_to(opp).rank() as f32;
            opp_best_passer = opp_best_passer.max((r / 6.0).min(1.0));
        }
    }
    v[12] = stm_best_passer;
    v[13] = opp_best_passer;
    v[14] = if stm_best_passer >= 0.83 { 1.0 } else { 0.0 }; // 6th/7th rank passer
    v[15] = if opp_best_passer >= 0.83 { 1.0 } else { 0.0 };

    // 16..24: Fortress & Endgame Draw Detectors (opposite-colored bishops, blocked pawns, low material)
    let wb = pos.piece_bb(Color::White, Piece::Bishop);
    let bb = pos.piece_bb(Color::Black, Piece::Bishop);
    let opp_colored_bishops = wb.count() == 1
        && bb.count() == 1
        && ((wb & Bitboard::LIGHT_SQUARES).non_empty() != (bb & Bitboard::LIGHT_SQUARES).non_empty());
    let no_majors = (pos.piece_bb(Color::White, Piece::Queen)
        | pos.piece_bb(Color::Black, Piece::Queen)
        | pos.piece_bb(Color::White, Piece::Rook)
        | pos.piece_bb(Color::Black, Piece::Rook))
    .is_empty();
    let wp = pos.piece_bb(Color::White, Piece::Pawn);
    let bp = pos.piece_bb(Color::Black, Piece::Pawn);
    let blocked_pawns = (wp.north() & bp).count() as f32;
    let total_pawns = (wp.count() + bp.count()).max(1) as f32;

    v[16] = if opp_colored_bishops { 1.0 } else { 0.0 };
    v[17] = if opp_colored_bishops && no_majors { 1.0 } else { 0.0 };
    v[18] = (blocked_pawns * 2.0 / total_pawns).min(1.0); // pawn chain lock ratio
    v[19] = (pos.halfmove_clock as f32 / 100.0).min(1.0);
    v[20] = if pos.is_insufficient_material() { 1.0 } else { 0.0 };
    v[21] = (1.0 - (pos.total_piece_count() as f32 / 32.0)).clamp(0.0, 1.0); // endgame factor
    v[22] = (pos.material_balance_stm() as f32 / 1000.0).clamp(-1.0, 1.0);
    v[23] = if no_majors { 1.0 } else { 0.0 };

    // 24..32: Ray-battery & piece balance ratios
    let stm_heavy = (pos.piece_bb(stm, Piece::Queen).count() * 2 + pos.piece_bb(stm, Piece::Rook).count()) as f32;
    let opp_heavy = (pos.piece_bb(opp, Piece::Queen).count() * 2 + pos.piece_bb(opp, Piece::Rook).count()) as f32;
    let stm_minor = (pos.piece_bb(stm, Piece::Bishop).count() + pos.piece_bb(stm, Piece::Knight).count()) as f32;
    let opp_minor = (pos.piece_bb(opp, Piece::Bishop).count() + pos.piece_bb(opp, Piece::Knight).count()) as f32;
    v[24] = (stm_heavy / 4.0).min(1.0);
    v[25] = (opp_heavy / 4.0).min(1.0);
    v[26] = (stm_minor / 4.0).min(1.0);
    v[27] = (opp_minor / 4.0).min(1.0);
    v[28] = (pawn_attacks(stm, stm_ksq) & pos.piece_bb(opp, Piece::Pawn)).count() as f32 * 0.5;
    v[29] = (knight_attacks(opp_ksq) & pos.piece_bb(stm, Piece::Knight)).count() as f32 * 0.5;
    let open_files_near_opp_k = (0..8)
        .filter(|&d| (ray_bb(opp_ksq, d) & (wp | bp)).is_empty())
        .count() as f32
        / 8.0;
    v[30] = open_files_near_opp_k;
    let open_files_near_stm_k = (0..8)
        .filter(|&d| (ray_bb(stm_ksq, d) & (wp | bp)).is_empty())
        .count() as f32
        / 8.0;
    v[31] = open_files_near_stm_k;

    v
}

/// Computes STM and Opponent 64-bit threat masks for the ThreatMap64 auxiliary training target.
pub fn extract_threat_masks(pos: &Position) -> (u64, u64) {
    let stm = pos.side_to_move;
    let opp = stm.flip();
    let stm_atk = pos.attack_map(stm);
    let opp_atk = pos.attack_map(opp);

    // Orient bitboards to STM perspective
    let orient = |bb: Bitboard| -> u64 {
        if stm == Color::White {
            bb.0
        } else {
            bb.0.swap_bytes()
        }
    };
    (orient(stm_atk), orient(opp_atk))
}

#[allow(dead_code)]
fn _unused_slider_ref(sq: Square, occ: Bitboard) -> Bitboard {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}
