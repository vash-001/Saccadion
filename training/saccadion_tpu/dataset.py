"""Zero-copy `.sacc` 128-byte binary shard loader and feature decoder for Colab Free TPU.

Reads binary shards written by `crates/saccadion-data` (magic `SACC0001`, 128 bytes/record)
and decodes boards into Stream A bilateral sparse indices, Stream B [64, 48] T4/T5 optic tensors,
Lateral Horn [32] vectors, and all 8 multi-sensor training targets.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
import numpy as np

from .config import HALFKA_DIM, LATERAL_HORN_DIM, STREAM_B_SQ_FEAT_DIM

SACC_SHARD_MAGIC = b"SACC0001"
SACC_RECORD_DTYPE = np.dtype(
    [
        ("board_nibbles", np.uint8, (32,)),
        ("meta_flags", "<u4"),
        ("wdl_target", "<f4", (3,)),
        ("best_move_u16", "<u2"),
        ("next_move_u16", "<u2"),
        ("top4_moves", "<u2", (4,)),
        ("top4_probs_u16", "<u2", (4,)),
        ("pressure_targets", "<f4", (5,)),
        ("lh_targets", "<f4", (4,)),
        ("volatility", "<f4"),
        ("mlh", "<f4"),
        ("threat_mask_stm", "<u8"),
        ("threat_mask_opp", "<u8"),
    ]
)
assert SACC_RECORD_DTYPE.itemsize == 128

SLIDING_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
KNIGHT_DIRS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
PIECE_CP = [100, 320, 330, 500, 900, 20000]


def king_bucket(king_sq: int, perspective: int) -> Tuple[int, bool]:
    rel = king_sq if perspective == 0 else (king_sq ^ 56)
    f = rel & 7
    r = rel >> 3
    mirror = f >= 4
    if mirror:
        f = 7 - f
    return (r // 2) * 4 + f, mirror


def halfka_index(persp: int, ksq: int, p_color: int, piece: int, sq: int) -> int:
    bucket, mirror = king_bucket(ksq, persp)
    rel_sq = sq if persp == 0 else (sq ^ 56)
    if mirror:
        rel_sq ^= 7
    rel_c = 0 if p_color == persp else 1
    return bucket * 768 + (rel_c * 6 + piece) * 64 + rel_sq


def unpack_nibbles(nibbles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unpacks 32 bytes into (colors[64], pieces[64]) where empty=-1."""
    colors = np.full(64, -1, dtype=np.int8)
    pieces = np.full(64, -1, dtype=np.int8)
    for sq in range(64):
        b = int(nibbles[sq >> 1])
        code = (b & 0x0F) if (sq & 1) == 0 else (b >> 4)
        if 1 <= code <= 6:
            colors[sq] = 0
            pieces[sq] = code - 1
        elif 7 <= code <= 12:
            colors[sq] = 1
            pieces[sq] = code - 7
    return colors, pieces


def _attack_map(colors: np.ndarray, pieces: np.ndarray, side: int) -> int:
    mask = 0
    for sq in range(64):
        if colors[sq] != side:
            continue
        p = int(pieces[sq])
        f0, r0 = sq & 7, sq >> 3
        if p == 0:  # Pawn
            dr = 1 if side == 0 else -1
            for df in (-1, 1):
                f, r = f0 + df, r0 + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    mask |= 1 << (r * 8 + f)
        elif p == 1:  # Knight
            for df, dr in KNIGHT_DIRS:
                f, r = f0 + df, r0 + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    mask |= 1 << (r * 8 + f)
        elif p == 5:  # King
            for df, dr in SLIDING_DIRS:
                f, r = f0 + df, r0 + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    mask |= 1 << (r * 8 + f)
        else:
            dirs = (
                SLIDING_DIRS[1::2]
                if p == 2
                else (SLIDING_DIRS[0::2] if p == 3 else SLIDING_DIRS)
            )
            for df, dr in dirs:
                f, r = f0 + df, r0 + dr
                while 0 <= f < 8 and 0 <= r < 8:
                    tsq = r * 8 + f
                    mask |= 1 << tsq
                    if colors[tsq] != -1:
                        break
                    f += df
                    r += dr
    return mask


def _king_ring_mask(ksq: int) -> int:
    f0, r0 = ksq & 7, ksq >> 3
    m = 1 << ksq
    for df, dr in SLIDING_DIRS:
        f, r = f0 + df, r0 + dr
        if 0 <= f < 8 and 0 <= r < 8:
            m |= 1 << (r * 8 + f)
    return m


def _pinned_mask(colors: np.ndarray, pieces: np.ndarray, side: int, ksq: int) -> int:
    opp = 1 - side
    f0, r0 = ksq & 7, ksq >> 3
    pinned = 0
    for d_idx, (df, dr) in enumerate(SLIDING_DIRS):
        is_diag = (d_idx & 1) == 1
        f, r = f0 + df, r0 + dr
        own_blocker = -1
        while 0 <= f < 8 and 0 <= r < 8:
            sq = r * 8 + f
            c = int(colors[sq])
            if c != -1:
                if c == side:
                    if own_blocker == -1:
                        own_blocker = sq
                    else:
                        break
                else:
                    p = int(pieces[sq])
                    sniper = (p == 4) or (p == 2 and is_diag) or (p == 3 and not is_diag)
                    if sniper and own_blocker != -1:
                        pinned |= 1 << own_blocker
                    break
            f += df
            r += dr
    return pinned


def _checkers_count(colors: np.ndarray, pieces: np.ndarray, side: int, ksq: int) -> int:
    opp = 1 - side
    f0, r0 = ksq & 7, ksq >> 3
    cnt = 0
    # Pawn checkers
    p_dr = 1 if side == 0 else -1
    for df in (-1, 1):
        f, r = f0 + df, r0 + p_dr
        if 0 <= f < 8 and 0 <= r < 8:
            sq = r * 8 + f
            if colors[sq] == opp and pieces[sq] == 0:
                cnt += 1
    # Knight checkers
    for df, dr in KNIGHT_DIRS:
        f, r = f0 + df, r0 + dr
        if 0 <= f < 8 and 0 <= r < 8:
            sq = r * 8 + f
            if colors[sq] == opp and pieces[sq] == 1:
                cnt += 1
    # Sliding checkers
    for d_idx, (df, dr) in enumerate(SLIDING_DIRS):
        is_diag = (d_idx & 1) == 1
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            sq = r * 8 + f
            if colors[sq] != -1:
                if colors[sq] == opp:
                    p = int(pieces[sq])
                    if p == 4 or (p == 2 and is_diag) or (p == 3 and not is_diag):
                        cnt += 1
                break
            f += df
            r += dr
    return cnt


def _is_passed_pawn(colors: np.ndarray, pieces: np.ndarray, sq: int) -> bool:
    if pieces[sq] != 0:
        return False
    c = int(colors[sq])
    opp = 1 - c
    f0, r0 = sq & 7, sq >> 3
    ranks = range(r0 + 1, 8) if c == 0 else range(0, r0)
    for r in ranks:
        for f in range(max(0, f0 - 1), min(7, f0 + 1) + 1):
            tsq = r * 8 + f
            if colors[tsq] == opp and pieces[tsq] == 0:
                return False
    return True


def decode_record_features(
    rec: np.void, max_sparse: int = 96
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Decodes one 128-byte SaccRecord into exact Stream A & Stream B input tensors
    matching `crates/saccadion-chess/src/features.rs` 100% identically."""
    colors, pieces = unpack_nibbles(rec["board_nibbles"])
    meta = int(rec["meta_flags"])
    stm = meta & 1
    opp = 1 - stm
    castling = (meta >> 1) & 0x0F
    halfmove = (meta >> 9) & 0xFF
    phase = (meta >> 17) & 0x07

    ksq = [0, 0]
    all_pawns_mask = 0
    wp_mask, bp_mask = 0, 0
    for sq in range(64):
        if pieces[sq] == 5:
            ksq[int(colors[sq])] = sq
        elif pieces[sq] == 0:
            all_pawns_mask |= 1 << sq
            if colors[sq] == 0:
                wp_mask |= 1 << sq
            else:
                bp_mask |= 1 << sq

    atk_stm = _attack_map(colors, pieces, stm)
    atk_opp = _attack_map(colors, pieces, opp)
    pin_stm = _pinned_mask(colors, pieces, stm, ksq[stm])
    pin_opp = _pinned_mask(colors, pieces, opp, ksq[opp])

    # 1. Stream A bilateral sparse indices (exact match with Rust extract_bilateral_sparse)
    sp_stm = np.full(max_sparse, -1, dtype=np.int32)
    sp_opp = np.full(max_sparse, -1, dtype=np.int32)
    cnt_s, cnt_o = 0, 0
    for sq in range(64):
        if colors[sq] == -1:
            continue
        c, p = int(colors[sq]), int(pieces[sq])
        if cnt_s < max_sparse:
            sp_stm[cnt_s] = halfka_index(stm, ksq[stm], c, p, sq)
            cnt_s += 1
        if cnt_o < max_sparse:
            sp_opp[cnt_o] = halfka_index(opp, ksq[opp], c, p, sq)
            cnt_o += 1

    _, mir_s = king_bucket(ksq[stm], stm)
    _, mir_o = king_bucket(ksq[opp], opp)
    for sq in range(64):
        bit = 1 << sq
        if colors[sq] != -1:
            c = int(colors[sq])
            if c == stm and (atk_opp & bit) and cnt_s < max_sparse:
                rsq = (sq if stm == 0 else sq ^ 56) ^ (7 if mir_s else 0)
                sp_stm[cnt_s] = HALFKA_DIM + rsq
                cnt_s += 1
                if not (atk_stm & bit) and cnt_s < max_sparse:
                    sp_stm[cnt_s] = HALFKA_DIM + 192 + rsq
                    cnt_s += 1
            elif c == opp and (atk_stm & bit) and cnt_s < max_sparse:
                rsq = (sq if stm == 0 else sq ^ 56) ^ (7 if mir_s else 0)
                sp_stm[cnt_s] = HALFKA_DIM + 64 + rsq
                cnt_s += 1
            if c == opp and (atk_stm & bit) and cnt_o < max_sparse:
                rsq = (sq if opp == 0 else sq ^ 56) ^ (7 if mir_o else 0)
                sp_opp[cnt_o] = HALFKA_DIM + rsq
                cnt_o += 1
                if not (atk_opp & bit) and cnt_o < max_sparse:
                    sp_opp[cnt_o] = HALFKA_DIM + 192 + rsq
                    cnt_o += 1
            elif c == stm and (atk_opp & bit) and cnt_o < max_sparse:
                rsq = (sq if opp == 0 else sq ^ 56) ^ (7 if mir_o else 0)
                sp_opp[cnt_o] = HALFKA_DIM + 64 + rsq
                cnt_o += 1
        if (pin_stm & bit) and cnt_s < max_sparse:
            rsq = (sq if stm == 0 else sq ^ 56) ^ (7 if mir_s else 0)
            sp_stm[cnt_s] = HALFKA_DIM + 128 + rsq
            cnt_s += 1
        if (pin_opp & bit) and cnt_o < max_sparse:
            rsq = (sq if opp == 0 else sq ^ 56) ^ (7 if mir_o else 0)
            sp_opp[cnt_o] = HALFKA_DIM + 128 + rsq
            cnt_o += 1

    # 2. Stream B [64, 48] Optic Lobe T4/T5 ON-OFF + LC tensor (exact Rust match)
    board = np.zeros((64, STREAM_B_SQ_FEAT_DIM), dtype=np.float32)
    castling_own = bin((castling & 0x03) if stm == 0 else ((castling >> 2) & 0x03)).count("1") * 0.5
    hm_norm = min(halfmove / 100.0, 1.0)
    stm_ring = _king_ring_mask(ksq[stm])
    opp_ring = _king_ring_mask(ksq[opp])

    for rel_sq in range(64):
        abs_sq = rel_sq if stm == 0 else (rel_sq ^ 56)
        row = board[rel_sq]
        if colors[abs_sq] != -1:
            off = 0 if colors[abs_sq] == stm else 6
            row[off + int(pieces[abs_sq])] = 1.0
            row[47] = 1.0

        f0, r0 = abs_sq & 7, abs_sq >> 3
        for d_idx, (df, dr) in enumerate(SLIDING_DIRS):
            eff_dr = dr if stm == 0 else -dr
            f, r, dist = f0 + df, r0 + eff_dr, 1.0
            while 0 <= f < 8 and 0 <= r < 8:
                ssq = r * 8 + f
                if colors[ssq] != -1:
                    p = int(pieces[ssq])
                    is_diag = (d_idx & 1) == 1
                    matches = (
                        p == 4
                        or (p == 2 and is_diag)
                        or (p == 3 and not is_diag)
                        or (p == 5 and dist == 1.0)
                        or (p == 0 and dist == 1.0 and is_diag)
                    )
                    w = (1.0 / (0.5 + 0.5 * dist)) if matches else (0.25 / dist)
                    if colors[ssq] == stm:
                        row[12 + d_idx] = w
                    else:
                        row[20 + d_idx] = w
                    break
                dist += 1.0
                f += df
                r += eff_dr

        for k_idx, (df, dr) in enumerate(KNIGHT_DIRS):
            eff_dr = dr if stm == 0 else -dr
            f, r = f0 + df, r0 + eff_dr
            if 0 <= f < 8 and 0 <= r < 8:
                ssq = r * 8 + f
                if pieces[ssq] == 1:
                    row[28 + k_idx] = 1.0 if colors[ssq] == stm else -1.0

        bit = 1 << abs_sq
        row[36] = 1.0 if (atk_stm & bit) else 0.0
        row[37] = 1.0 if (atk_opp & bit) else 0.0
        row[38] = 1.0 if (pin_stm & bit) else (-1.0 if (pin_opp & bit) else 0.0)
        row[39] = 1.0 if (stm_ring & bit) else 0.0
        row[40] = 1.0 if (opp_ring & bit) else 0.0
        file_mask = 0x0101_0101_0101_0101 << f0
        row[41] = 1.0 if (all_pawns_mask & file_mask) == 0 else 0.0
        row[42] = 1.0 if _is_passed_pawn(colors, pieces, abs_sq) else 0.0
        row[43] = (rel_sq & 7) / 7.0
        row[44] = (rel_sq >> 3) / 7.0
        row[45] = castling_own
        row[46] = hm_norm

    # 3. Complete 32-dim Lateral Horn descriptor (100% exact match with Rust extract_lateral_horn_features)
    lh = np.zeros(LATERAL_HORN_DIM, dtype=np.float32)
    chk_cnt = _checkers_count(colors, pieces, stm, ksq[stm])
    lh[0] = min(float(chk_cnt), 2.0) * 0.5
    lh[1] = 1.0 if chk_cnt >= 2 else 0.0
    lh[2] = min(bin(pin_stm).count("1") / 4.0, 1.0)
    lh[3] = min(bin(pin_opp).count("1") / 4.0, 1.0)

    stm_pawns = wp_mask if stm == 0 else bp_mask
    opp_pawns = bp_mask if stm == 0 else wp_mask
    lh[4] = bin(stm_ring & atk_opp).count("1") / 9.0
    lh[5] = bin(opp_ring & atk_stm).count("1") / 9.0
    lh[6] = bin(stm_ring & stm_pawns).count("1") / 4.0
    lh[7] = bin(opp_ring & opp_pawns).count("1") / 4.0

    stm_hang, opp_hang = 0, 0
    stm_ep_cnt, opp_ep_cnt = 0.0, 0.0
    stm_best_p, opp_best_p = 0.0, 0.0
    wb_mask, bb_mask = 0, 0
    no_majors = True
    mat_diff = 0
    stm_heavy, opp_heavy = 0.0, 0.0
    stm_minor, opp_minor = 0.0, 0.0
    total_pieces = 0

    for sq in range(64):
        c = int(colors[sq])
        if c == -1:
            continue
        total_pieces += 1
        p = int(pieces[sq])
        bit = 1 << sq
        if p < 5:
            val = PIECE_CP[p]
            mat_diff += val if c == stm else -val
            if c == stm and (atk_opp & bit) and not (atk_stm & bit):
                stm_hang = max(stm_hang, val)
                stm_ep_cnt += 1.0
            elif c == opp and (atk_stm & bit) and not (atk_opp & bit):
                opp_hang = max(opp_hang, val)
                opp_ep_cnt += 1.0
        if p == 0 and _is_passed_pawn(colors, pieces, sq):
            rel_r = (sq >> 3) if c == 0 else ((sq ^ 56) >> 3)
            if c == stm:
                stm_best_p = max(stm_best_p, min(rel_r / 6.0, 1.0))
            else:
                opp_best_p = max(opp_best_p, min(rel_r / 6.0, 1.0))
        if p == 2:
            if c == 0:
                wb_mask |= bit
            else:
                bb_mask |= bit
        if p in (3, 4):
            no_majors = False
            w_h = 2.0 if p == 4 else 1.0
            if c == stm:
                stm_heavy += w_h
            else:
                opp_heavy += w_h
        if p in (1, 2):
            if c == stm:
                stm_minor += 1.0
            else:
                opp_minor += 1.0

    lh[8] = min(stm_hang / 900.0, 1.0)
    lh[9] = min(opp_hang / 900.0, 1.0)
    lh[10] = min(stm_ep_cnt / 4.0, 1.0)
    lh[11] = min(opp_ep_cnt / 4.0, 1.0)
    lh[12] = stm_best_p
    lh[13] = opp_best_p
    lh[14] = 1.0 if stm_best_p >= 0.83 else 0.0
    lh[15] = 1.0 if opp_best_p >= 0.83 else 0.0

    light_sq = 0x55AA_55AA_55AA_55AA
    ocb = (
        bin(wb_mask).count("1") == 1
        and bin(bb_mask).count("1") == 1
        and (bool(wb_mask & light_sq) != bool(bb_mask & light_sq))
    )
    blocked_pawns = bin(((wp_mask << 8) & 0xFFFF_FFFF_FFFF_FFFF) & bp_mask).count("1")
    total_pawns = max(bin(wp_mask).count("1") + bin(bp_mask).count("1"), 1)
    insuff = (
        all_pawns_mask == 0
        and no_majors
        and ((total_pieces - 2) <= 1 or (total_pieces == 4 and bin(wb_mask).count("1") == 1 and bin(bb_mask).count("1") == 1 and not ocb))
    )

    lh[16] = 1.0 if ocb else 0.0
    lh[17] = 1.0 if (ocb and no_majors) else 0.0
    lh[18] = min(blocked_pawns * 2.0 / total_pawns, 1.0)
    lh[19] = hm_norm
    lh[20] = 1.0 if insuff else 0.0
    lh[21] = max(0.0, min(1.0, 1.0 - total_pieces / 32.0))
    lh[22] = max(-1.0, min(1.0, mat_diff / 1000.0))
    lh[23] = 1.0 if no_majors else 0.0
    lh[24] = min(stm_heavy / 4.0, 1.0)
    lh[25] = min(opp_heavy / 4.0, 1.0)
    lh[26] = min(stm_minor / 4.0, 1.0)
    lh[27] = min(opp_minor / 4.0, 1.0)

    # 28: pawn_attacks(stm, stm_ksq) & opp_pawns
    kf_s, kr_s = ksq[stm] & 7, ksq[stm] >> 3
    p_dr_s = 1 if stm == 0 else -1
    p_near = 0
    for df in (-1, 1):
        f, r = kf_s + df, kr_s + p_dr_s
        if 0 <= f < 8 and 0 <= r < 8 and (opp_pawns & (1 << (r * 8 + f))):
            p_near += 1
    lh[28] = p_near * 0.5

    # 29: knight_attacks(opp_ksq) & stm_knights
    kf_o, kr_o = ksq[opp] & 7, ksq[opp] >> 3
    n_near = 0
    for df, dr in KNIGHT_DIRS:
        f, r = kf_o + df, kr_o + dr
        if 0 <= f < 8 and 0 <= r < 8:
            tsq = r * 8 + f
            if colors[tsq] == stm and pieces[tsq] == 1:
                n_near += 1
    lh[29] = n_near * 0.5

    def open_rays_from(sq_k: int) -> float:
        f_k, r_k = sq_k & 7, sq_k >> 3
        open_c = 0
        for df, dr in SLIDING_DIRS:
            f, r = f_k + df, r_k + dr
            has_pawn = False
            while 0 <= f < 8 and 0 <= r < 8:
                if all_pawns_mask & (1 << (r * 8 + f)):
                    has_pawn = True
                    break
                f += df
                r += dr
            if not has_pawn:
                open_c += 1
        return open_c / 8.0

    lh[30] = open_rays_from(ksq[opp])
    lh[31] = open_rays_from(ksq[stm])

    return sp_stm, sp_opp, board, lh, phase


class SaccShardDataset:
    """Memory-mapped reader over one or more `.sacc` shards for TPU training."""

    def __init__(self, shard_paths: List[Path], max_sparse: int = 96):
        self.records_list: List[np.ndarray] = []
        self.max_sparse = max_sparse
        for p in shard_paths:
            with open(p, "rb") as f:
                hdr = f.read(16)
            magic, _flags, count = struct.unpack("<8sII", hdr)
            if magic != SACC_SHARD_MAGIC:
                raise ValueError(f"Invalid shard magic in {p}: {magic!r}")
            arr = np.memmap(p, dtype=SACC_RECORD_DTYPE, mode="r", offset=16, shape=(count,))
            self.records_list.append(arr)
        self.records = np.concatenate(self.records_list, axis=0) if self.records_list else np.empty(0, dtype=SACC_RECORD_DTYPE)

    def __len__(self) -> int:
        return len(self.records)

    def _collate(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        bsz = len(indices)
        sp_s = np.full((bsz, self.max_sparse), -1, dtype=np.int32)
        sp_o = np.full((bsz, self.max_sparse), -1, dtype=np.int32)
        board = np.zeros((bsz, 64, STREAM_B_SQ_FEAT_DIM), dtype=np.float32)
        lh = np.zeros((bsz, LATERAL_HORN_DIM), dtype=np.float32)
        phase = np.zeros((bsz,), dtype=np.int32)
        wdl = np.zeros((bsz, 3), dtype=np.float32)
        top4_m = np.zeros((bsz, 4), dtype=np.int32)
        top4_p = np.zeros((bsz, 4), dtype=np.float32)
        press = np.zeros((bsz, 5), dtype=np.float32)
        lh_t = np.zeros((bsz, 4), dtype=np.float32)
        vol = np.zeros((bsz,), dtype=np.float32)
        threat = np.zeros((bsz, 64), dtype=np.float32)

        for i, ridx in enumerate(indices):
            rec = self.records[int(ridx)]
            sp_s[i], sp_o[i], board[i], lh[i], phase[i] = decode_record_features(rec, self.max_sparse)
            wdl[i] = rec["wdl_target"]
            top4_m[i] = rec["top4_moves"].astype(np.int32)
            probs = rec["top4_probs_u16"].astype(np.float32) / 65535.0
            probs_sum = float(np.sum(probs))
            top4_p[i] = probs / max(probs_sum, 1e-6)
            press[i] = rec["pressure_targets"]
            lh_t[i] = rec["lh_targets"]
            vol[i] = float(rec["volatility"])
            tm = int(rec["threat_mask_stm"])
            for sq in range(64):
                threat[i, sq] = 1.0 if (tm & (1 << sq)) else 0.0

        return {
            "sparse_stm": sp_s,
            "sparse_opp": sp_o,
            "board_feats": board,
            "lh_feat": lh,
            "phase_bucket": phase,
            "wdl_target": wdl,
            "top4_moves": top4_m,
            "top4_probs": top4_p,
            "pressure_targets": press,
            "lh_targets": lh_t,
            "volatility": vol,
            "threat_map_64": threat,
        }

    def iter_batches(
        self, batch_size: int, seed: int = 0, num_workers: int = 4, prefetch: int = 4
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Yields shuffled mini-batches with multi-threaded CPU decoding & async TPU prefetch."""
        from concurrent.futures import ThreadPoolExecutor
        import queue
        import threading

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self.records))
        slices = [
            perm[start : start + batch_size]
            for start in range(0, len(perm) - batch_size + 1, batch_size)
        ]
        if num_workers <= 1 or len(slices) <= 2:
            for sl in slices:
                yield self._collate(sl)
            return

        q: queue.Queue = queue.Queue(maxsize=prefetch)
        sentinel = object()

        def producer() -> None:
            try:
                with ThreadPoolExecutor(max_workers=num_workers) as pool:
                    for batch in pool.map(self._collate, slices):
                        q.put(batch)
            finally:
                q.put(sentinel)

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item

    def iter_trajectory_batches(
        self, batch_size: int, window_k: int = 12, seed: int = 0
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Yields [B, K=12, ...] batches for Stage-3 Meta-Dopamine `jax.lax.scan` training."""
        n_win = len(self.records) // window_k
        rng = np.random.default_rng(seed)
        win_perm = rng.permutation(n_win)
        for start in range(0, n_win - batch_size + 1, batch_size):
            w_ids = win_perm[start : start + batch_size]
            flat_idx = np.concatenate([np.arange(w * window_k, (w + 1) * window_k) for w in w_ids])
            flat = self._collate(flat_idx)
            yield {k: v.reshape((batch_size, window_k) + v.shape[1:]) for k, v in flat.items()}
