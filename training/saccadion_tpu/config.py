"""Saccadion-1 architectural configuration matching Rust `crates/saccadion-nn/src/config.rs` byte-for-byte."""

from dataclasses import dataclass
import numpy as np

SACCNET_MAGIC = b"SACCNET1"
SACCNET_VERSION = 6
OPTIC_LAYERS = 3
PLASTIC_RANK = 8
AMACRINE_RANK = 8
GRAPH_EDGE_TYPES = 4

HALFKA_DIM = 16 * 768
RAY_SPARSE_DIM = 256 + 1024
STREAM_A_SPARSE_DIM = HALFKA_DIM + RAY_SPARSE_DIM  # 13,568
STREAM_B_SQ_FEAT_DIM = 48
LATERAL_HORN_DIM = 32

# 16 Elite Causal Chess Concepts (AOTU -> LAL -> SMP Effectome Curriculum)
# Curated strictly for deep strategic/causal understanding without diluting training.
CAUSAL_CONCEPTS: tuple[str, ...] = (
    "quietMove",
    "zugzwang",
    "sacrifice",
    "clearance",
    "interference",
    "deflection",
    "overloading",
    "attraction",
    "discoveredAttack",
    "pin",
    "trappedPiece",
    "defensiveMove",
    "exposedKing",
    "kingsideAttack",
    "queensideAttack",
    "advancedPawn",
)
NUM_CAUSAL_CONCEPTS = len(CAUSAL_CONCEPTS)
CAUSAL_CONCEPT_TO_IDX = {name: i for i, name in enumerate(CAUSAL_CONCEPTS)}


@dataclass(frozen=True)
class SaccadionConfig:
    tier_id: int
    sparse_input_dim: int = STREAM_A_SPARSE_DIM
    acc_dim: int = 1024
    flyhash_claws: int = 8
    flyhash_seed: int = 0x5ACC_AD10_F174_A541
    mbon_dim: int = 32
    num_phase_stacks: int = 8
    lh_input_dim: int = LATERAL_HORN_DIM
    lh_hidden_dim: int = 32
    sq_feat_dim: int = STREAM_B_SQ_FEAT_DIM
    stream_b_dim: int = 128
    num_integrators: int = 16
    num_broadcasters: int = 16
    cx_sectors: int = 16
    cx_dim: int = 64
    cx_max_ticks: int = 6
    num_creative_plans: int = 4
    seed_rank: int = 32

    @classmethod
    def tiny(cls) -> "SaccadionConfig":
        return cls(
            tier_id=0,
            acc_dim=64,
            stream_b_dim=32,
            num_integrators=8,
            num_broadcasters=8,
            cx_dim=16,
            cx_max_ticks=4,
            seed_rank=16,
        )

    @classmethod
    def nano(cls) -> "SaccadionConfig":
        return cls(
            tier_id=1,
            acc_dim=512,
            stream_b_dim=64,
            num_integrators=16,
            num_broadcasters=16,
            cx_dim=32,
            cx_max_ticks=6,
            seed_rank=32,
        )

    @classmethod
    def pro(cls) -> "SaccadionConfig":
        return cls(
            tier_id=2,
            acc_dim=1024,
            stream_b_dim=128,
            num_integrators=16,
            num_broadcasters=16,
            cx_dim=64,
            cx_max_ticks=6,
            seed_rank=32,
        )

    def lobe_dims_per_hemi(self) -> tuple[int, int, int]:
        ab = (self.acc_dim * 3) // 8
        gamma = self.acc_dim // 2
        apbp = self.acc_dim - ab - gamma
        return ab, gamma, apbp

    def plastic_dim_per_hemi(self) -> int:
        _, gamma, apbp = self.lobe_dims_per_hemi()
        return gamma + apbp


def _splitmix64(state: int) -> tuple[int, int]:
    """Exact 64-bit SplitMix64 matching Rust `config::splitmix64` bit-for-bit."""
    mask = 0xFFFF_FFFF_FFFF_FFFF
    state = (state + 0x9E37_79B9_7F4A_7C15) & mask
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & mask
    z = (z ^ (z >> 31)) & mask
    return state, z


def build_flyhash_claws(cfg: SaccadionConfig) -> np.ndarray:
    """Deterministic FlyHash sparse claw index matrix [plastic_dim, flyhash_claws]."""
    plastic_n = cfg.plastic_dim_per_hemi()
    claws = cfg.flyhash_claws
    out = np.zeros((plastic_n, claws), dtype=np.int32)
    rng = cfg.flyhash_seed
    for p_i in range(plastic_n):
        chosen = []
        while len(chosen) < claws:
            rng, val = _splitmix64(rng)
            cand = int(val % cfg.acc_dim)
            if cand not in chosen:
                chosen.append(cand)
        out[p_i] = chosen
    return out


def init_sez_innate_weights(sparse_input_dim: int, num_phase_stacks: int) -> np.ndarray:
    """Deterministic Drosophila SEZ (Subesophageal Zone) innate survival valence prior (in pawns).

    Matches Rust `crates/saccadion-nn/src/weights.rs::init_sez_innate_weights` bit-for-bit!
    """
    w = np.zeros((sparse_input_dim, num_phase_stacks), dtype=np.float32)
    piece_pawns = (1.00, 3.20, 3.30, 5.00, 9.00, 0.00)
    halfka_dim = 16 * 768
    for bucket in range(16):
        for rel_c in range(2):
            sign = 1.0 if rel_c == 0 else -1.0
            for piece in range(6):
                base_v = piece_pawns[piece]
                for rel_sq in range(64):
                    f = bucket * 768 + (rel_c * 6 + piece) * 64 + rel_sq
                    if f >= halfka_dim or f >= sparse_input_dim:
                        continue
                    file_i = float(rel_sq & 7)
                    rank_i = float(rel_sq >> 3)
                    center = 1.0 - (abs(file_i - 3.5) + abs(rank_i - 3.5)) / 7.0
                    if piece == 0:
                        psq_bonus = 0.06 * rank_i + 0.04 * center
                    elif piece in (1, 2):
                        psq_bonus = 0.14 * center
                    elif piece == 3:
                        psq_bonus = 0.04 * center
                    elif piece == 4:
                        psq_bonus = 0.05 * center
                    else:
                        psq_bonus = 0.0
                    w[f, :] = np.float32(sign * (base_v + psq_bonus))
    return w
