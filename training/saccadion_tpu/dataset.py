"""Zero-copy `.sacc` 128-byte binary shard loader and feature decoder for Colab Free TPU.

Reads binary shards written by `crates/saccadion-data` (magic `SACC0001`, 128 bytes/record)
and decodes boards into Stream A bilateral sparse indices, Stream B [64, 48] T4/T5 optic tensors,
Lateral Horn [32] vectors, and all 8 multi-sensor training targets.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import struct
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import numpy as np

from .config import HALFKA_DIM, LATERAL_HORN_DIM, NUM_CAUSAL_CONCEPTS, STREAM_B_SQ_FEAT_DIM

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


def _piece_attacks_from(colors: np.ndarray, side: int, p: int, sq: int) -> int:
    """Exact 64-bit attack bitmask for a piece (side, p) on `sq`, matching Rust `piece_attacks_from`."""
    mask = 0
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


def _attack_map(colors: np.ndarray, pieces: np.ndarray, side: int) -> int:
    mask = 0
    for sq in range(64):
        if colors[sq] != side:
            continue
        mask |= _piece_attacks_from(colors, side, int(pieces[sq]), sq)
    return mask


def _piece_xray_attacks_from(colors: np.ndarray, p: int, sq: int) -> int:
    """Exact 64-bit X-Ray Through-Blocker bitmask for sliders (Bishop=2, Rook=3, Queen=4).
    Matches Rust `slider_attacks(sq, all_occ & !first_atks) & !first_atks` bit-for-bit!"""
    if p not in (2, 3, 4):
        return 0
    mask = 0
    f0, r0 = sq & 7, sq >> 3
    dirs = (
        SLIDING_DIRS[1::2]
        if p == 2
        else (SLIDING_DIRS[0::2] if p == 3 else SLIDING_DIRS)
    )
    for df, dr in dirs:
        f, r = f0 + df, r0 + dr
        blockers = 0
        while 0 <= f < 8 and 0 <= r < 8:
            tsq = r * 8 + f
            if blockers == 1:
                mask |= 1 << tsq
            if colors[tsq] != -1:
                blockers += 1
                if blockers >= 2:
                    break
            f += df
            r += dr
    return mask


def extract_chess_graph_adj_py(colors: np.ndarray, pieces: np.ndarray, stm: int) -> np.ndarray:
    """Exact 6-channel STM-oriented FlyWire X-Ray Chess-Graph Adjacency [6, 64, 2] uint32 (lo/hi).

    Channels: 0=STM Atk, 1=STM Def, 2=Opp Atk, 3=Opp Def, 4=STM X-Ray, 5=Opp X-Ray.
    Matches Rust `crates/saccadion-chess/src/features.rs::extract_chess_graph_adj` bit-for-bit!
    """
    adj = np.zeros((6, 64, 2), dtype=np.uint32)
    stm_occ = 0
    opp_occ = 0
    for sq in range(64):
        c = int(colors[sq])
        if c == stm:
            stm_occ |= 1 << sq
        elif c != -1:
            opp_occ |= 1 << sq

    def orient_bb(bb: int) -> int:
        return bb if stm == 0 else int.from_bytes(bb.to_bytes(8, "little")[::-1], "little")

    for sq in range(64):
        c = int(colors[sq])
        if c == -1:
            continue
        p = int(pieces[sq])
        atks = _piece_attacks_from(colors, c, p, sq)
        xray = _piece_xray_attacks_from(colors, p, sq)
        rel_u = sq if stm == 0 else (sq ^ 56)
        if c == stm:
            b0 = orient_bb(atks & (~stm_occ & 0xFFFF_FFFF_FFFF_FFFF))
            b1 = orient_bb(atks & stm_occ)
            b4 = orient_bb(xray)
            adj[0, rel_u, 0] = b0 & 0xFFFF_FFFF
            adj[0, rel_u, 1] = (b0 >> 32) & 0xFFFF_FFFF
            adj[1, rel_u, 0] = b1 & 0xFFFF_FFFF
            adj[1, rel_u, 1] = (b1 >> 32) & 0xFFFF_FFFF
            adj[4, rel_u, 0] = b4 & 0xFFFF_FFFF
            adj[4, rel_u, 1] = (b4 >> 32) & 0xFFFF_FFFF
        else:
            b2 = orient_bb(atks & (~opp_occ & 0xFFFF_FFFF_FFFF_FFFF))
            b3 = orient_bb(atks & opp_occ)
            b5 = orient_bb(xray)
            adj[2, rel_u, 0] = b2 & 0xFFFF_FFFF
            adj[2, rel_u, 1] = (b2 >> 32) & 0xFFFF_FFFF
            adj[3, rel_u, 0] = b3 & 0xFFFF_FFFF
            adj[3, rel_u, 1] = (b3 >> 32) & 0xFFFF_FFFF
            adj[5, rel_u, 0] = b5 & 0xFFFF_FFFF
            adj[5, rel_u, 1] = (b5 >> 32) & 0xFFFF_FFFF
    return adj


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Decodes one 128-byte SaccRecord into exact Stream A & Stream B input tensors
    (including LC4/LPLC2 pairwise edges + FlyWire Chess-Graph adjacency)
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

    # 1c. LC4/LPLC2 Pairwise Threat-Edge Graph sparse features (exact Rust match!)
    all_occ = 0
    for sq in range(64):
        if colors[sq] != -1:
            all_occ |= 1 << sq
    seen_s = set()
    seen_o = set()
    for c in (0, 1):
        for p in range(6):
            for u in range(64):
                if colors[u] != c or pieces[u] != p:
                    continue
                atks = _piece_attacks_from(colors, c, p, u) & all_occ
                m = atks
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    vc, vp = int(colors[v]), int(pieces[v])
                    is_attack = vc != c
                    # STM hemisphere
                    if cnt_s < max_sparse:
                        rv_s = (v if stm == 0 else v ^ 56) ^ (7 if mir_s else 0)
                        quad_s = ((rv_s >> 5) << 1) | ((rv_s & 7) >> 2)
                        ec_s = (0 if c == stm else 2) + (0 if is_attack else 1)
                        lc4_s = (ec_s * 256 + (((p * 6 + vp) * 4 + quad_s) & 255)) & 1023
                        if lc4_s not in seen_s:
                            seen_s.add(lc4_s)
                            sp_stm[cnt_s] = HALFKA_DIM + 256 + lc4_s
                            cnt_s += 1
                    # Opponent hemisphere
                    if cnt_o < max_sparse:
                        rv_o = (v if opp == 0 else v ^ 56) ^ (7 if mir_o else 0)
                        quad_o = ((rv_o >> 5) << 1) | ((rv_o & 7) >> 2)
                        ec_o = (0 if c == opp else 2) + (0 if is_attack else 1)
                        lc4_o = (ec_o * 256 + (((p * 6 + vp) * 4 + quad_o) & 255)) & 1023
                        if lc4_o not in seen_o:
                            seen_o.add(lc4_o)
                            sp_opp[cnt_o] = HALFKA_DIM + 256 + lc4_o
                            cnt_o += 1

    # 1d. FlyWire Typed Directed Chess-Graph Adjacency [4, 64, 2] uint32
    graph_adj = extract_chess_graph_adj_py(colors, pieces, stm)

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

    return sp_stm, sp_opp, board, graph_adj, lh, phase


MAX_SPARSE_CACHE = 96
_BIT_SHIFTS_16 = np.arange(NUM_CAUSAL_CONCEPTS, dtype=np.uint16)


def _load_optional_concepts(shard_path: Path, count: int) -> np.ndarray:
    cpath = shard_path.with_name(shard_path.name + ".concepts.u16")
    if cpath.exists() and cpath.stat().st_size == count * 2:
        return np.fromfile(cpath, dtype="<u2", count=count)
    return np.zeros((count,), dtype=np.uint16)


def _load_optional_contrast(shard_path: Path, count: int) -> np.ndarray:
    cpath = shard_path.with_name(shard_path.name + ".contrast.u32")
    if cpath.exists() and cpath.stat().st_size == count * 4:
        return np.fromfile(cpath, dtype="<u4", count=count)
    return np.zeros((count,), dtype=np.uint32)


def _unpack_contrast_u32(cu: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpacks oriented contrast u32 -> (blunder_move int32, blunder_drop_cp float32, is_brilliant float32)."""
    bmv = (cu & np.uint32(0x0FFF)).astype(np.int32)
    is_bril = ((cu >> np.uint32(12)) & np.uint32(1)).astype(np.float32)
    drop_cp = ((cu >> np.uint32(16)) & np.uint32(0xFFFF)).astype(np.float32)
    return bmv, drop_cp, is_bril


def _decode_chunk_worker(
    args: Tuple[bytes, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Top-level multiprocessing worker bypassing Python GIL across all CPU cores."""
    raw_bytes, max_sparse = args
    recs = np.frombuffer(raw_bytes, dtype=SACC_RECORD_DTYPE)
    n = len(recs)
    sp_s = np.full((n, max_sparse), -1, dtype=np.int16)
    sp_o = np.full((n, max_sparse), -1, dtype=np.int16)
    bd = np.zeros((n, 64, STREAM_B_SQ_FEAT_DIM), dtype=np.float16)
    ga = np.zeros((n, 6, 64, 2), dtype=np.uint32)
    lh = np.zeros((n, LATERAL_HORN_DIM), dtype=np.float32)
    ph = np.zeros((n,), dtype=np.uint8)
    for i in range(n):
        s_i, o_i, b_i, g_i, l_i, p_i = decode_record_features(recs[i], max_sparse)
        sp_s[i] = s_i.astype(np.int16)
        sp_o[i] = o_i.astype(np.int16)
        bd[i] = b_i.astype(np.float16)
        ga[i] = g_i
        lh[i] = l_i
        ph[i] = p_i
    return sp_s, sp_o, bd, ga, lh, ph


def _find_rust_data_binary() -> Optional[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    for cand in [
        repo_root / "target" / "release" / "saccadion-data",
        repo_root / "target" / "release" / "saccadion-data.exe",
        Path("/content/drive/MyDrive/Saccadion_TPU/bin_v14/saccadion-data"),
        Path("/content/drive/MyDrive/Saccadion_TPU/bin_v13/saccadion-data"),
        Path("/content/drive/MyDrive/Saccadion_TPU/bin_v12/saccadion-data"),
        Path("/content/drive/MyDrive/Saccadion_TPU/bin_v11/saccadion-data"),
        Path("/content/drive/MyDrive/Saccadion_TPU/bin_v10/saccadion-data"),
        Path("/content/drive/MyDrive/Saccadion_TPU/bin_v9/saccadion-data"),
    ]:
        if cand.exists():
            return cand
    return None


def _shards_fingerprint(shard_paths: List[Path]) -> str:
    h = hashlib.sha256()
    for p in shard_paths:
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}|".encode())
        cp = p.with_name(p.name + ".concepts.u16")
        if cp.exists():
            cst = cp.stat()
            h.update(f"c:{cst.st_size}:{int(cst.st_mtime)}|".encode())
        kp = p.with_name(p.name + ".contrast.u32")
        if kp.exists():
            kst = kp.stat()
            h.update(f"k:{kst.st_size}:{int(kst.st_mtime)}|".encode())
    return h.hexdigest()[:16]


def _orient_move_u16_vec(moves: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """Flips (from_sq, to_sq) vertically (sq ^ 56) when stm == 1 (Black) so all policy,
    Noduli, and VNC-2Ply targets match Stream B's STM-oriented board perspective 100%."""
    m = moves.astype(np.int32)
    mask = (stm != 0) & (m != 0)
    u = ((m >> 6) & 63) ^ 56
    v = (m & 63) ^ 56
    flipped = (u << 6) | v
    return np.where(mask, flipped, m).astype(np.uint16)


def _orient_move_u16_scalar(mv: int, stm: int) -> int:
    if stm == 0 or mv == 0:
        return mv
    return ((((mv >> 6) & 63) ^ 56) << 6) | ((mv & 63) ^ 56)


def build_or_open_memmap_cache(
    shard_paths: List[Path],
    cache_root: Path,
    split_name: str,
    max_sparse: int = MAX_SPARSE_CACHE,
) -> Optional[Path]:
    """Builds (ONCE) or opens a zero-copy flat binary memmap cache for 50x faster TPU feeding!

    Uses Pure-Rust multi-threaded `saccadion-data predecode` (~1.5s for 1M positions!) when
    available, with a multi-process Python `ProcessPoolExecutor` fallback (bypasses GIL).
    """
    if not shard_paths:
        return None
    fp = _shards_fingerprint(shard_paths)
    cdir = cache_root / f"{split_name}_v8_{fp}"
    meta_file = cdir / "meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("version") == 8 and int(meta.get("count", 0)) > 0:
                return cdir
        except Exception:
            pass

    cdir.mkdir(parents=True, exist_ok=True)
    rust_bin = _find_rust_data_binary()
    if rust_bin is not None:
        print(f"[FastCache v8] Rust multi-threaded INT8+6ch-XRay pre-decoding `{split_name}` ({len(shard_paths)} shards)...")
        res = subprocess.run(
            [str(rust_bin), "predecode", str(cdir)] + [str(p) for p in shard_paths],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and meta_file.exists():
            try:
                m_chk = json.loads(meta_file.read_text(encoding="utf-8"))
                if m_chk.get("version") == 8:
                    print(" ", res.stdout.strip())
                    return cdir
            except Exception:
                pass
        print(f"[Warn] Rust predecode unavailable/outdated ({res.stderr.strip()}); using Python multiprocessing.")

    # Multi-process Python fallback (bypasses GIL across all CPU cores!)
    t0 = time.time()
    recs_list: List[np.ndarray] = []
    concepts_list: List[np.ndarray] = []
    contrast_list: List[np.ndarray] = []
    for p in shard_paths:
        with open(p, "rb") as f:
            hdr = f.read(16)
        magic, _flags, count = struct.unpack("<8sII", hdr)
        if magic != SACC_SHARD_MAGIC:
            raise ValueError(f"Invalid shard magic in {p}: {magic!r}")
        arr = np.memmap(p, dtype=SACC_RECORD_DTYPE, mode="r", offset=16, shape=(count,))
        recs_list.append(arr)
        concepts_list.append(_load_optional_concepts(p, count))
        contrast_list.append(_load_optional_contrast(p, count))

    all_recs = np.concatenate(recs_list, axis=0)
    all_concepts = np.concatenate(concepts_list, axis=0)
    all_contrast = np.concatenate(contrast_list, axis=0)
    n = len(all_recs)
    # Global cross-shard shuffle prevents contiguous puzzle/survival blocks from skewing epochs!
    if len(shard_paths) > 1 and n > 1:
        g_perm = np.random.default_rng(202605).permutation(n)
        all_recs = all_recs[g_perm]
        all_concepts = all_concepts[g_perm]
        all_contrast = all_contrast[g_perm]
    n_workers = max(1, min(os.cpu_count() or 2, 8))
    chunk_sz = max(512, min(8192, (n + n_workers * 4 - 1) // (n_workers * 4)))
    print(
        f"[FastCache] Building one-time local SSD/RAM cache for `{split_name}` "
        f"({n:,} pos across {n_workers} CPU processes)..."
    )

    sp_s = np.memmap(cdir / "sparse_stm.i16", dtype="<i2", mode="w+", shape=(n, max_sparse))
    sp_o = np.memmap(cdir / "sparse_opp.i16", dtype="<i2", mode="w+", shape=(n, max_sparse))
    bd = np.memmap(cdir / "board_feats.i8", dtype=np.int8, mode="w+", shape=(n, 64, STREAM_B_SQ_FEAT_DIM))
    ga = np.memmap(cdir / "graph_adj.u32", dtype="<u4", mode="w+", shape=(n, 6, 64, 2))
    lh = np.memmap(cdir / "lh_feat.f32", dtype="<f4", mode="w+", shape=(n, LATERAL_HORN_DIM))
    ph = np.memmap(cdir / "phase.u8", dtype=np.uint8, mode="w+", shape=(n,))

    tasks = [
        (all_recs[start : min(n, start + chunk_sz)].tobytes(), max_sparse)
        for start in range(0, n, chunk_sz)
    ]
    offset = 0
    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for s_c, o_c, b_c, g_c, l_c, p_c in pool.map(_decode_chunk_worker, tasks):
                m = len(p_c)
                sp_s[offset : offset + m] = s_c
                sp_o[offset : offset + m] = o_c
                bd[offset : offset + m] = np.clip(np.rint(b_c.astype(np.float32) * 127.0), -127, 127).astype(np.int8)
                ga[offset : offset + m] = g_c
                lh[offset : offset + m] = l_c
                ph[offset : offset + m] = p_c
                offset += m
    else:
        for t_arg in tasks:
            s_c, o_c, b_c, g_c, l_c, p_c = _decode_chunk_worker(t_arg)
            m = len(p_c)
            sp_s[offset : offset + m] = s_c
            sp_o[offset : offset + m] = o_c
            bd[offset : offset + m] = np.clip(np.rint(b_c.astype(np.float32) * 127.0), -127, 127).astype(np.int8)
            ga[offset : offset + m] = g_c
            lh[offset : offset + m] = l_c
            ph[offset : offset + m] = p_c
            offset += m

    sp_s.flush()
    sp_o.flush()
    bd.flush()
    ga.flush()
    lh.flush()
    ph.flush()

    # Vectorized extraction of target arrays with exact STM move orientation!
    stm_arr = (all_recs["meta_flags"] & np.uint32(1)).astype(np.int32)
    np.ascontiguousarray(all_recs["wdl_target"], dtype="<f4").tofile(cdir / "wdl.f32")
    top4_oriented = _orient_move_u16_vec(all_recs["top4_moves"], stm_arr[:, None])
    np.ascontiguousarray(top4_oriented, dtype="<u2").tofile(cdir / "top4_moves.u16")
    probs = all_recs["top4_probs_u16"].astype(np.float32) / 65535.0
    probs_sum = np.maximum(np.sum(probs, axis=-1, keepdims=True), 1e-6)
    np.ascontiguousarray(probs / probs_sum, dtype="<f4").tofile(cdir / "top4_probs.f32")
    next_oriented = _orient_move_u16_vec(all_recs["next_move_u16"], stm_arr)
    np.ascontiguousarray(next_oriented, dtype="<u2").tofile(cdir / "next_move.u16")
    np.ascontiguousarray(all_recs["pressure_targets"], dtype="<f4").tofile(cdir / "pressure.f32")
    np.ascontiguousarray(all_recs["lh_targets"], dtype="<f4").tofile(cdir / "lh_targets.f32")
    np.ascontiguousarray(all_recs["volatility"], dtype="<f4").tofile(cdir / "vol.f32")
    # Vectorized 64-bit threat_mask_stm unpack -> [N, 64] uint8
    tm = all_recs["threat_mask_stm"][:, None]
    shifts = np.arange(64, dtype=np.uint64)[None, :]
    threat_u8 = ((tm >> shifts) & np.uint64(1)).astype(np.uint8)
    threat_u8.tofile(cdir / "threat_64.u8")
    np.ascontiguousarray(all_concepts, dtype="<u2").tofile(cdir / "causal_bits.u16")
    bmv_raw = (all_contrast & np.uint32(0x0FFF)).astype(np.uint16)
    bmv_ori = _orient_move_u16_vec(bmv_raw, stm_arr).astype(np.uint32)
    contrast_ori = (all_contrast & np.uint32(0xFFFF_F000)) | (bmv_ori & np.uint32(0x0FFF))
    np.ascontiguousarray(contrast_ori, dtype="<u4").tofile(cdir / "contrast.u32")

    meta_file.write_text(
        json.dumps({"count": n, "max_sparse": max_sparse, "version": 8, "num_shards": len(shard_paths)}),
        encoding="utf-8",
    )
    dt = max(1e-3, time.time() - t0)
    print(f"  [FastCache] Built `{split_name}` cache ({n:,} pos) in {dt:.1f}s ({n/dt:,.0f} pos/s) -> {cdir}")
    return cdir


class SaccShardDataset:
    """High-speed memory-mapped reader over `.sacc` shards with automatic 50x FastCache support."""

    def __init__(
        self,
        shard_paths: List[Path],
        max_sparse: int = MAX_SPARSE_CACHE,
        cache_dir: Optional[Path] = None,
        split_name: str = "train",
    ):
        self.shard_paths = list(shard_paths)
        self.max_sparse = max_sparse
        self._cached = False
        self._cache_path: Optional[Path] = None

        if cache_dir is not None and shard_paths:
            cdir = build_or_open_memmap_cache(shard_paths, cache_dir, split_name, max_sparse=max_sparse)
            if cdir is not None and (cdir / "meta.json").exists():
                self._open_cache(cdir)
                return

        self.records_list: List[np.ndarray] = []
        self.concepts_list: List[np.ndarray] = []
        self.contrast_list: List[np.ndarray] = []
        for p in shard_paths:
            with open(p, "rb") as f:
                hdr = f.read(16)
            magic, _flags, count = struct.unpack("<8sII", hdr)
            if magic != SACC_SHARD_MAGIC:
                raise ValueError(f"Invalid shard magic in {p}: {magic!r}")
            arr = np.memmap(p, dtype=SACC_RECORD_DTYPE, mode="r", offset=16, shape=(count,))
            self.records_list.append(arr)
            self.concepts_list.append(_load_optional_concepts(p, count))
            self.contrast_list.append(_load_optional_contrast(p, count))
        self.records = (
            np.concatenate(self.records_list, axis=0)
            if self.records_list
            else np.empty(0, dtype=SACC_RECORD_DTYPE)
        )
        self.concepts = (
            np.concatenate(self.concepts_list, axis=0)
            if self.concepts_list
            else np.empty(0, dtype=np.uint16)
        )
        self.contrast = (
            np.concatenate(self.contrast_list, axis=0)
            if self.contrast_list
            else np.empty(0, dtype=np.uint32)
        )
        self._count = len(self.records)

    def _open_cache(self, cdir: Path) -> None:
        meta = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))
        n = int(meta["count"])
        ms = int(meta["max_sparse"])
        self._count = n
        self.max_sparse = ms
        self._cached = True
        self._cache_path = cdir
        self._sp_s = np.memmap(cdir / "sparse_stm.i16", dtype="<i2", mode="r", shape=(n, ms))
        self._sp_o = np.memmap(cdir / "sparse_opp.i16", dtype="<i2", mode="r", shape=(n, ms))
        bd_i8 = cdir / "board_feats.i8"
        if bd_i8.exists():
            self._bd_is_i8 = True
            self._bd = np.memmap(bd_i8, dtype=np.int8, mode="r", shape=(n, 64, STREAM_B_SQ_FEAT_DIM))
        else:
            self._bd_is_i8 = False
            self._bd = np.memmap(cdir / "board_feats.f16", dtype="<f2", mode="r", shape=(n, 64, STREAM_B_SQ_FEAT_DIM))
        g_ch = 6 if int(meta.get("version", 7)) >= 8 else 4
        self._ga = np.memmap(cdir / "graph_adj.u32", dtype="<u4", mode="r", shape=(n, g_ch, 64, 2))
        self._lh = np.memmap(cdir / "lh_feat.f32", dtype="<f4", mode="r", shape=(n, LATERAL_HORN_DIM))
        self._ph = np.memmap(cdir / "phase.u8", dtype=np.uint8, mode="r", shape=(n,))
        self._wdl = np.memmap(cdir / "wdl.f32", dtype="<f4", mode="r", shape=(n, 3))
        self._t4m = np.memmap(cdir / "top4_moves.u16", dtype="<u2", mode="r", shape=(n, 4))
        self._t4p = np.memmap(cdir / "top4_probs.f32", dtype="<f4", mode="r", shape=(n, 4))
        self._nm = np.memmap(cdir / "next_move.u16", dtype="<u2", mode="r", shape=(n,))
        self._pr = np.memmap(cdir / "pressure.f32", dtype="<f4", mode="r", shape=(n, 5))
        self._lht = np.memmap(cdir / "lh_targets.f32", dtype="<f4", mode="r", shape=(n, 4))
        self._vol = np.memmap(cdir / "vol.f32", dtype="<f4", mode="r", shape=(n,))
        self._th = np.memmap(cdir / "threat_64.u8", dtype=np.uint8, mode="r", shape=(n, 64))
        self._cb = np.memmap(cdir / "causal_bits.u16", dtype="<u2", mode="r", shape=(n,))
        cpath = cdir / "contrast.u32"
        self._ct = np.memmap(cpath, dtype="<u4", mode="r", shape=(n,)) if cpath.exists() else np.zeros((n,), dtype=np.uint32)

    def __len__(self) -> int:
        return self._count

    def _collate(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        if self._cached:
            cb = np.asarray(self._cb[indices], dtype=np.uint16)
            causal_concepts = ((cb[:, None] >> _BIT_SHIFTS_16[None, :]) & np.uint16(1)).astype(np.float32)
            causal_mask = (cb > 0).astype(np.float32)
            bmv, bdrop, is_bril = _unpack_contrast_u32(np.asarray(self._ct[indices], dtype=np.uint32))
            bd_arr = np.asarray(self._bd[indices], dtype=np.float32)
            if getattr(self, "_bd_is_i8", False):
                bd_arr = bd_arr * (1.0 / 127.0)
            return {
                "sparse_stm": np.asarray(self._sp_s[indices], dtype=np.int32),
                "sparse_opp": np.asarray(self._sp_o[indices], dtype=np.int32),
                "board_feats": bd_arr,
                "graph_adj": np.asarray(self._ga[indices], dtype=np.uint32),
                "lh_feat": np.asarray(self._lh[indices], dtype=np.float32),
                "phase_bucket": np.asarray(self._ph[indices], dtype=np.int32),
                "wdl_target": np.asarray(self._wdl[indices], dtype=np.float32),
                "top4_moves": np.asarray(self._t4m[indices], dtype=np.int32),
                "top4_probs": np.asarray(self._t4p[indices], dtype=np.float32),
                "next_move": np.asarray(self._nm[indices], dtype=np.int32),
                "pressure_targets": np.asarray(self._pr[indices], dtype=np.float32),
                "lh_targets": np.asarray(self._lht[indices], dtype=np.float32),
                "volatility": np.asarray(self._vol[indices], dtype=np.float32),
                "threat_map_64": np.asarray(self._th[indices], dtype=np.float32),
                "causal_concepts": causal_concepts,
                "causal_mask": causal_mask,
                "blunder_move": bmv,
                "blunder_drop_cp": bdrop,
                "is_brilliant": is_bril,
            }

        bsz = len(indices)
        sp_s = np.full((bsz, self.max_sparse), -1, dtype=np.int32)
        sp_o = np.full((bsz, self.max_sparse), -1, dtype=np.int32)
        board = np.zeros((bsz, 64, STREAM_B_SQ_FEAT_DIM), dtype=np.float32)
        gadj = np.zeros((bsz, 6, 64, 2), dtype=np.uint32)
        lh = np.zeros((bsz, LATERAL_HORN_DIM), dtype=np.float32)
        phase = np.zeros((bsz,), dtype=np.int32)
        wdl = np.zeros((bsz, 3), dtype=np.float32)
        top4_m = np.zeros((bsz, 4), dtype=np.int32)
        top4_p = np.zeros((bsz, 4), dtype=np.float32)
        next_m = np.zeros((bsz,), dtype=np.int32)
        press = np.zeros((bsz, 5), dtype=np.float32)
        lh_t = np.zeros((bsz, 4), dtype=np.float32)
        vol = np.zeros((bsz,), dtype=np.float32)
        threat = np.zeros((bsz, 64), dtype=np.float32)
        cb = np.zeros((bsz,), dtype=np.uint16)
        ct = np.zeros((bsz,), dtype=np.uint32)

        for i, ridx in enumerate(indices):
            ii = int(ridx)
            rec = self.records[ii]
            stm_i = int(rec["meta_flags"]) & 1
            sp_s[i], sp_o[i], board[i], gadj[i], lh[i], phase[i] = decode_record_features(rec, self.max_sparse)
            wdl[i] = rec["wdl_target"]
            top4_m[i] = [_orient_move_u16_scalar(int(m), stm_i) for m in rec["top4_moves"]]
            probs = rec["top4_probs_u16"].astype(np.float32) / 65535.0
            probs_sum = float(np.sum(probs))
            top4_p[i] = probs / max(probs_sum, 1e-6)
            next_m[i] = _orient_move_u16_scalar(int(rec["next_move_u16"]), stm_i)
            press[i] = rec["pressure_targets"]
            lh_t[i] = rec["lh_targets"]
            vol[i] = float(rec["volatility"])
            tm = int(rec["threat_mask_stm"])
            for sq in range(64):
                threat[i, sq] = 1.0 if (tm & (1 << sq)) else 0.0
            cb[i] = self.concepts[ii]
            c_raw = int(self.contrast[ii])
            bmv_ori = _orient_move_u16_scalar(c_raw & 0x0FFF, stm_i)
            ct[i] = (c_raw & 0xFFFF_F000) | (bmv_ori & 0x0FFF)

        causal_concepts = ((cb[:, None] >> _BIT_SHIFTS_16[None, :]) & np.uint16(1)).astype(np.float32)
        causal_mask = (cb > 0).astype(np.float32)
        bmv, bdrop, is_bril = _unpack_contrast_u32(ct)
        return {
            "sparse_stm": sp_s,
            "sparse_opp": sp_o,
            "board_feats": board,
            "graph_adj": gadj,
            "lh_feat": lh,
            "phase_bucket": phase,
            "wdl_target": wdl,
            "top4_moves": top4_m,
            "top4_probs": top4_p,
            "next_move": next_m,
            "pressure_targets": press,
            "lh_targets": lh_t,
            "volatility": vol,
            "threat_map_64": threat,
            "causal_concepts": causal_concepts,
            "causal_mask": causal_mask,
            "blunder_move": bmv,
            "blunder_drop_cp": bdrop,
            "is_brilliant": is_bril,
        }

    def _load_contiguous_chunk(self, start: int, end: int) -> Dict[str, np.ndarray]:
        """Reads a contiguous slice [start:end] sequentially from NVMe/RAM in native compact dtypes.

        Keeps `board_feats` in native `float16` (2x less RAM & 2x faster Host->TPU transfer; cast to
        f32/bf16 takes <0.01ms on TPU!) to completely eliminate multi-second CPU stalls.
        """
        cb = np.asarray(self._cb[start:end], dtype=np.uint16)
        causal_concepts = ((cb[:, None] >> _BIT_SHIFTS_16[None, :]) & np.uint16(1)).astype(np.float32)
        causal_mask = (cb > 0).astype(np.float32)
        bmv, bdrop, is_bril = _unpack_contrast_u32(np.asarray(self._ct[start:end], dtype=np.uint32))
        return {
            "sparse_stm": np.asarray(self._sp_s[start:end], dtype=np.int32),
            "sparse_opp": np.asarray(self._sp_o[start:end], dtype=np.int32),
            "board_feats": np.array(
                self._bd[start:end],
                dtype=np.int8 if getattr(self, "_bd_is_i8", False) else np.float16,
                copy=True,
            ),
            "graph_adj": np.array(self._ga[start:end], dtype=np.uint32, copy=True),
            "lh_feat": np.asarray(self._lh[start:end], dtype=np.float32),
            "phase_bucket": np.asarray(self._ph[start:end], dtype=np.int32),
            "wdl_target": np.asarray(self._wdl[start:end], dtype=np.float32),
            "top4_moves": np.asarray(self._t4m[start:end], dtype=np.int32),
            "top4_probs": np.asarray(self._t4p[start:end], dtype=np.float32),
            "next_move": np.asarray(self._nm[start:end], dtype=np.int32),
            "pressure_targets": np.asarray(self._pr[start:end], dtype=np.float32),
            "lh_targets": np.asarray(self._lht[start:end], dtype=np.float32),
            "volatility": np.asarray(self._vol[start:end], dtype=np.float32),
            "threat_map_64": np.asarray(self._th[start:end], dtype=np.float32),
            "causal_concepts": causal_concepts,
            "causal_mask": causal_mask,
            "blunder_move": bmv,
            "blunder_drop_cp": bdrop,
            "is_brilliant": is_bril,
        }

    def iter_batches(
        self,
        batch_size: int,
        seed: int = 0,
        prefetch: int = 12,
        mirror_prob: float = 0.0,
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Yields shuffled mini-batches via a 2-stage overlapped async pipeline (ZERO TPU stalls!).

        Stage 1 (`disk_reader`): pre-reads raw compact f16 blocks from NVMe/RAM in background.
        Stage 2 (`batch_builder`): shuffles, slices, and applies exact horizontal mirror symmetry
            in background, keeping a deep `prefetch=12` queue permanently full for the TPU!
        """
        rng = np.random.default_rng(seed)

        if self._cached and self._count >= batch_size * 2:
            # Globally pre-shuffled super-blocks (16,384 recs = 32 batches, mix_k=1 -> ZERO np.concatenate copy
            # and maximum Linux sequential SSD read-ahead! Only ~48 MB per block in INT8!)
            block_sz = max(batch_size * 4, 16384)
            blocks = [
                (s, min(self._count, s + block_sz))
                for s in range(0, self._count, block_sz)
            ]
            block_order = rng.permutation(len(blocks))
            mix_k = 1

            chunk_q: queue.Queue = queue.Queue(maxsize=2)
            batch_q: queue.Queue = queue.Queue(maxsize=max(4, prefetch))
            stop_evt = threading.Event()
            disk_done = threading.Event()
            build_done = threading.Event()
            sentinel = object()

            def _put_safe(target_q: queue.Queue, item: object) -> bool:
                while not stop_evt.is_set():
                    try:
                        target_q.put(item, timeout=0.05)
                        return True
                    except queue.Full:
                        continue
                return False

            def disk_reader() -> None:
                try:
                    for bi in range(0, len(block_order), mix_k):
                        if stop_evt.is_set():
                            break
                        chosen = block_order[bi : bi + mix_k]
                        chunks = [self._load_contiguous_chunk(*blocks[int(idx)]) for idx in chosen]
                        if len(chunks) == 1:
                            merged = chunks[0]
                        else:
                            merged = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
                        del chunks
                        if not _put_safe(chunk_q, merged):
                            del merged
                            break
                finally:
                    disk_done.set()
                    _put_safe(chunk_q, sentinel)

            def batch_builder() -> None:
                aug_rng = np.random.default_rng(seed * 7919 + 13)
                try:
                    while not stop_evt.is_set():
                        try:
                            merged = chunk_q.get(timeout=0.05)
                        except queue.Empty:
                            if disk_done.is_set():
                                break
                            continue
                        if merged is sentinel:
                            break
                        m_len = len(merged["phase_bucket"])
                        local_perm = rng.permutation(m_len)
                        for pos in range(0, m_len - batch_size + 1, batch_size):
                            if stop_evt.is_set():
                                break
                            idx_sl = local_perm[pos : pos + batch_size]
                            batch = {k: v[idx_sl] for k, v in merged.items()}
                            if mirror_prob > 0.0:
                                batch = augment_batch_file_mirror(batch, aug_rng, prob=mirror_prob)
                            if not _put_safe(batch_q, batch):
                                break
                        del merged
                finally:
                    build_done.set()
                    _put_safe(batch_q, sentinel)

            t_disk = threading.Thread(target=disk_reader, daemon=True)
            t_build = threading.Thread(target=batch_builder, daemon=True)
            t_disk.start()
            t_build.start()
            try:
                while True:
                    try:
                        item = batch_q.get(timeout=0.1)
                    except queue.Empty:
                        if build_done.is_set():
                            break
                        continue
                    if item is sentinel:
                        break
                    yield item
            finally:
                stop_evt.set()
                for q_ref in (chunk_q, batch_q):
                    while not q_ref.empty():
                        try:
                            q_ref.get_nowait()
                        except queue.Empty:
                            break
                t_disk.join(timeout=1.5)
                t_build.join(timeout=1.5)
            return

        perm = rng.permutation(self._count)
        slices = [
            perm[start : start + batch_size]
            for start in range(0, len(perm) - batch_size + 1, batch_size)
        ]
        aug_rng = np.random.default_rng(seed * 7919 + 13)
        for sl in slices:
            batch = self._collate(sl)
            if mirror_prob > 0.0:
                batch = augment_batch_file_mirror(batch, aug_rng, prob=mirror_prob)
            yield batch

    def iter_trajectory_batches(
        self, batch_size: int, window_k: int = 12, seed: int = 0
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Yields [B, K=12, ...] batches for Stage-3 Meta-Dopamine `jax.lax.scan` training."""
        n_win = self._count // window_k
        rng = np.random.default_rng(seed)
        win_perm = rng.permutation(n_win)
        for start in range(0, n_win - batch_size + 1, batch_size):
            w_ids = win_perm[start : start + batch_size]
            flat_idx = np.concatenate([np.arange(w * window_k, (w + 1) * window_k) for w in w_ids])
            flat = self._collate(flat_idx)
            yield {k: v.reshape((batch_size, window_k) + v.shape[1:]) for k, v in flat.items()}


_SQ_MIRROR_IDX = np.array([sq ^ 7 for sq in range(64)], dtype=np.int32)
_SLIDE_MIRROR = np.array([0, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32)
_KNIGHT_MIRROR = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int32)
_BYTE_BITREV = np.array([int(f"{b:08b}"[::-1], 2) for b in range(256)], dtype=np.uint8)


def augment_batch_file_mirror(
    batch: Dict[str, np.ndarray], rng: np.random.Generator, prob: float = 0.5
) -> Dict[str, np.ndarray]:
    """Applies exact horizontal file-mirror symmetry (a <-> h) on castling-free positions.

    HalfKA, FullRays, and LC4 sparse indices are already horizontally mirror-invariant via
    king_bucket, so mirroring board_feats, FlyWire graph_adj (u -> u^7 AND v -> v^7),
    move targets, and threat_map_64 is 100% mathematically exact!
    """
    bd = batch["board_feats"]
    bsz = bd.shape[0]
    # Castling-free check: col 45 is castling_own (0.0 when STM has no castling rights)
    no_castle = bd[:, 0, 45] == 0.0
    do_flip = no_castle & (rng.random(bsz) < prob)
    if not np.any(do_flip):
        return batch

    out = dict(batch)
    bd_new = bd.copy()
    sub = bd_new[do_flip][:, _SQ_MIRROR_IDX, :]
    sub[:, :, 12:20] = sub[:, :, 12 + _SLIDE_MIRROR]
    sub[:, :, 20:28] = sub[:, :, 20 + _SLIDE_MIRROR]
    sub[:, :, 28:36] = sub[:, :, 28 + _KNIGHT_MIRROR]
    if sub.dtype == np.int8:
        sub[:, :, 43] = np.int8(127) - sub[:, :, 43]
    else:
        sub[:, :, 43] = sub.dtype.type(1.0) - sub[:, :, 43]
    bd_new[do_flip] = sub
    out["board_feats"] = bd_new

    if "graph_adj" in batch:
        ga_new = batch["graph_adj"].copy()
        ga_sub = np.ascontiguousarray(ga_new[do_flip][:, :, _SQ_MIRROR_IDX, :])
        ga_bytes = ga_sub.view(np.uint8)
        ga_bytes[...] = _BYTE_BITREV[ga_bytes]
        ga_new[do_flip] = ga_sub
        out["graph_adj"] = ga_new

    def _mirror_mv(m: np.ndarray) -> np.ndarray:
        nonzero = m != 0
        u = ((m >> 6) & 63) ^ 7
        v = (m & 63) ^ 7
        return np.where(nonzero, (u << 6) | v, m)

    t4 = batch["top4_moves"].copy()
    t4[do_flip] = _mirror_mv(t4[do_flip])
    out["top4_moves"] = t4

    nm = batch["next_move"].copy()
    nm[do_flip] = _mirror_mv(nm[do_flip])
    out["next_move"] = nm

    if "blunder_move" in batch:
        bm = batch["blunder_move"].copy()
        bm[do_flip] = _mirror_mv(bm[do_flip])
        out["blunder_move"] = bm

    th = batch["threat_map_64"].copy()
    th[do_flip] = th[do_flip][:, _SQ_MIRROR_IDX]
    out["threat_map_64"] = th
    return out
