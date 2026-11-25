//! Saccadion-1 architectural configuration shared byte-for-byte with JAX TPU training.

use saccadion_chess::features::{LATERAL_HORN_DIM, STREAM_A_SPARSE_DIM, STREAM_B_SQ_FEAT_DIM};

pub const SACCNET_MAGIC: [u8; 8] = *b"SACCNET1";
pub const SACCNET_VERSION: u32 = 1;

/// Exact architectural dimensions matching FlyWire v783 biological ratios.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelConfig {
    pub tier_id: u32,             // 0 = Tiny (test), 1 = Nano, 2 = Pro
    pub sparse_input_dim: usize,  // 12,544 (16*768 HalfKA + 256 FullRays)
    pub acc_dim: usize,           // Per-hemisphere accumulator / Kenyon Cell count
    pub flyhash_claws: usize,     // Sparse claw fan-in per gamma/apbp KC (Dasgupta 2017: ~6-8)
    pub flyhash_seed: u64,        // Deterministic PRNG seed for identical Rust/JAX claw wiring
    pub mbon_dim: usize,          // 32 Push-Pull MBONs (16 E + 16 I)
    pub num_phase_stacks: usize,  // 8 game-phase output stacks
    pub lh_input_dim: usize,      // 32 Lateral Horn tactical/fortress descriptors
    pub lh_hidden_dim: usize,     // 32 LHLN 98.2%-inhibitory hidden units
    pub sq_feat_dim: usize,       // 48 T4/T5 + LC input features per board square
    pub stream_b_dim: usize,      // Stream B hidden width D (Pro=128, Nano=64, Tiny=32)
    pub num_integrators: usize,   // Rich-Club Integrator hub count
    pub num_broadcasters: usize,  // Rich-Club Broadcaster hub count
    pub cx_sectors: usize,        // 16 Central Complex compass sectors (8 rays + 8 knights)
    pub cx_dim: usize,            // Phasor channel width per compass sector
    pub cx_max_ticks: usize,      // Max recurrent deliberation ticks (~4.42 FlyWire hops -> 6)
    pub num_creative_plans: usize,// K = 4 orthogonal strategic goal vectors
    pub seed_rank: usize,         // Low-rank Stream B -> Stream A fast-weight seed & mask rank
}

impl ModelConfig {
    /// Compact test tier for rapid local unit tests and exact JAX <-> Rust numerical parity.
    pub fn tiny() -> Self {
        Self {
            tier_id: 0,
            sparse_input_dim: STREAM_A_SPARSE_DIM,
            acc_dim: 64,
            flyhash_claws: 8,
            flyhash_seed: 0x5ACC_AD10_F174_A541,
            mbon_dim: 32,
            num_phase_stacks: 8,
            lh_input_dim: LATERAL_HORN_DIM,
            lh_hidden_dim: 32,
            sq_feat_dim: STREAM_B_SQ_FEAT_DIM,
            stream_b_dim: 32,
            num_integrators: 8,
            num_broadcasters: 8,
            cx_sectors: 16,
            cx_dim: 16,
            cx_max_ticks: 4,
            num_creative_plans: 4,
            seed_rank: 16,
        }
    }

    /// Rapid Colab Free TPU scientific ablation tier (~13.9M sparse + ~1.8M dense).
    pub fn nano() -> Self {
        Self {
            tier_id: 1,
            sparse_input_dim: STREAM_A_SPARSE_DIM,
            acc_dim: 512,
            flyhash_claws: 8,
            flyhash_seed: 0x5ACC_AD10_F174_A541,
            mbon_dim: 32,
            num_phase_stacks: 8,
            lh_input_dim: LATERAL_HORN_DIM,
            lh_hidden_dim: 32,
            sq_feat_dim: STREAM_B_SQ_FEAT_DIM,
            stream_b_dim: 64,
            num_integrators: 16,
            num_broadcasters: 16,
            cx_sectors: 16,
            cx_dim: 32,
            cx_max_ticks: 6,
            num_creative_plans: 4,
            seed_rank: 32,
        }
    }

    /// Full CPU Stockfish-18 challenger tier (~25-55M sparse + ~6M weight-tied recurrent dense).
    pub fn pro() -> Self {
        Self {
            tier_id: 2,
            sparse_input_dim: STREAM_A_SPARSE_DIM,
            acc_dim: 1024,
            flyhash_claws: 8,
            flyhash_seed: 0x5ACC_AD10_F174_A541,
            mbon_dim: 32,
            num_phase_stacks: 8,
            lh_input_dim: LATERAL_HORN_DIM,
            lh_hidden_dim: 32,
            sq_feat_dim: STREAM_B_SQ_FEAT_DIM,
            stream_b_dim: 128,
            num_integrators: 16,
            num_broadcasters: 16,
            cx_sectors: 16,
            cx_dim: 64,
            cx_max_ticks: 6,
            num_creative_plans: 4,
            seed_rank: 32,
        }
    }

    /// Exact FlyWire v783 three-lobe split per hemisphere (37.5% ab / 50.0% gamma / 12.5% apbp).
    #[inline(always)]
    pub fn lobe_dims_per_hemi(&self) -> (usize, usize, usize) {
        let ab = (self.acc_dim * 3) / 8;
        let gamma = self.acc_dim / 2;
        let apbp = self.acc_dim - ab - gamma;
        (ab, gamma, apbp)
    }

    #[inline(always)]
    pub fn plastic_dim_per_hemi(&self) -> usize {
        let (_, gamma, apbp) = self.lobe_dims_per_hemi();
        gamma + apbp
    }
}

/// Deterministic SplitMix64 PRNG shared identically between Rust and Python/JAX.
#[inline(always)]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Builds the deterministic FlyHash sparse claw wiring table for plastic Kenyon Cells
/// (gamma + alpha'/beta' lobes). Each plastic KC samples `flyhash_claws` distinct indices
/// from its own hemisphere accumulator (Dasgupta, Stevens & Navlakha, Science 2017).
pub fn build_flyhash_claws(cfg: &ModelConfig) -> Vec<u16> {
    let plastic_n = cfg.plastic_dim_per_hemi();
    let claws = cfg.flyhash_claws;
    let mut table = Vec::with_capacity(plastic_n * claws);
    let mut rng = cfg.flyhash_seed;
    for _ in 0..plastic_n {
        let mut chosen = [u16::MAX; 16];
        let mut c = 0usize;
        while c < claws {
            let cand = (splitmix64(&mut rng) as usize % cfg.acc_dim) as u16;
            if !chosen[..c].contains(&cand) {
                chosen[c] = cand;
                table.push(cand);
                c += 1;
            }
        }
    }
    table
}
