//! Complete Saccadion-1 weight storage and deterministic `.saccnet` binary serialization.
//!
//! Layout is shared identically with `training/saccadion_tpu/export_rust.py` so JAX-trained
//! checkpoints load directly into Pure Rust with bit-verified numerical parity.

use crate::config::{build_flyhash_claws, splitmix64, ModelConfig, SACCNET_MAGIC, SACCNET_VERSION};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// All static (frozen after TPU training) weights of Stream A + Stream B.
#[derive(Clone, Debug)]
pub struct SaccadionWeights {
    pub cfg: ModelConfig,
    /// Deterministic FlyHash sparse claw indices: [plastic_dim_per_hemi * flyhash_claws].
    pub flyhash_claws: Vec<u16>,

    // ==================== STREAM A (Fast Reflex & Calyx) ====================
    /// Sparse feature transformer: [sparse_input_dim * acc_dim] (row-major by feature index).
    pub a_ft_weight: Vec<f32>,
    pub a_ft_bias: Vec<f32>,
    /// FlyHash claw weights for plastic KCs: [plastic_dim_per_hemi * flyhash_claws].
    pub a_claw_weight: Vec<f32>,
    /// APL global GABA inhibition scaling per hemisphere [2] and per-KC thresholds [acc_dim].
    pub a_apl_beta: [f32; 2],
    pub a_apl_thresh: Vec<f32>,
    /// Push-Pull MBON projections (2:1 ipsi/contra) per game-phase stack:
    /// [num_phase_stacks * acc_dim * mbon_dim] for ipsi and contra.
    pub a_mbon_ipsi: Vec<f32>,
    pub a_mbon_contra: Vec<f32>,
    pub a_mbon_bias: Vec<f32>, // [num_phase_stacks * mbon_dim]
    /// Lateral Horn 98.2%-inhibitory innate veto branch:
    /// w_inh: [lh_hidden_dim * lh_input_dim], b_inh: [lh_hidden_dim],
    /// w_out: [4 * lh_hidden_dim], b_out: [4].
    pub a_lh_w_inh: Vec<f32>,
    pub a_lh_b_inh: Vec<f32>,
    pub a_lh_w_out: Vec<f32>,
    pub a_lh_b_out: [f32; 4],
    /// SMP/CRE/SIP Convergence Head per stack -> 3 WDL logits + scalar cp calibration:
    /// Input is [mbon_stm (32) || mbon_opp (32) || lh_veto (4)] = 68.
    pub a_smp_w: Vec<f32>, // [num_phase_stacks * 3 * (2 * mbon_dim + 4)]
    pub a_smp_b: Vec<f32>, // [num_phase_stacks * 3]

    // ==================== STREAM B (Deep Strategic Brain) ====================
    /// B1. T4/T5 Shunting Inhibition Optic Encoder:
    /// w_exc: [stream_b_dim * sq_feat_dim], w_inh: [stream_b_dim * sq_feat_dim],
    /// g_leak: [stream_b_dim], w_lc: [stream_b_dim * sq_feat_dim], b_optic: [stream_b_dim].
    pub b_shunt_exc: Vec<f32>,
    pub b_shunt_inh: Vec<f32>,
    pub b_shunt_leak: Vec<f32>,
    pub b_lc_w: Vec<f32>,
    pub b_optic_b: Vec<f32>,

    /// B2. Rich-Club Hub Core (Integrators + Broadcasters):
    pub b_hub_init: Vec<f32>,      // [(num_integrators + num_broadcasters) * stream_b_dim]
    pub b_int_q: Vec<f32>,         // [stream_b_dim * stream_b_dim]
    pub b_int_k: Vec<f32>,         // [stream_b_dim * stream_b_dim]
    pub b_int_v: Vec<f32>,         // [stream_b_dim * stream_b_dim]
    pub b_int_o: Vec<f32>,         // [stream_b_dim * stream_b_dim]
    pub b_rc_mix1: Vec<f32>,       // [stream_b_dim * stream_b_dim]
    pub b_rc_mix2: Vec<f32>,       // [stream_b_dim * stream_b_dim]
    pub b_rc_ln_g: Vec<f32>,       // [stream_b_dim]
    pub b_rc_ln_b: Vec<f32>,       // [stream_b_dim]

    /// B3. Central Complex Phasor Engine (weight-tied across T = 1..6 ticks):
    /// PFN force projection from board summary + integrators into 16 compass sectors x cx_dim.
    pub b_pfn_stm: Vec<f32>,       // [cx_sectors * cx_dim * stream_b_dim]
    pub b_pfn_opp: Vec<f32>,       // [cx_sectors * cx_dim * stream_b_dim]
    /// vDelta vertical composition & hDelta 180-deg phase subtraction: [cx_dim * cx_dim].
    pub b_vdelta: Vec<f32>,
    pub b_hdelta: Vec<f32>,
    /// Delta7 inhibitory ring sharpener: [cx_dim * cx_dim].
    pub b_delta7: Vec<f32>,
    /// CX Ring/Tangential context inhibition fed by [integrator_mean (D) || mbon (32)]:
    /// [cx_dim * (stream_b_dim + mbon_dim)].
    pub b_ring_inh: Vec<f32>,
    pub b_ring_b: Vec<f32>,        // [cx_dim]

    /// B4. Visual Centrifugal Top-Down Inhibitory Gate (71.8% I -> 64 squares) & Broadcaster back to board:
    pub b_cent_w: Vec<f32>,        // [64 * stream_b_dim]
    pub b_cent_b: Vec<f32>,        // [64]
    pub b_broad_to_sq: Vec<f32>,   // [stream_b_dim * stream_b_dim]
    pub b_sq_upd: Vec<f32>,        // [stream_b_dim * stream_b_dim]
    pub b_sq_ln_g: Vec<f32>,       // [stream_b_dim]
    pub b_sq_ln_b: Vec<f32>,       // [stream_b_dim]

    /// B5. K = 4 Creative Orthogonal Goal Plan Bank + PFL3 Policy + PFL2 Gain + Noduli + Pressure + Seed:
    pub b_plan_bank: Vec<f32>,     // [num_creative_plans * cx_dim * (cx_dim + stream_b_dim)]
    pub b_plan_b: Vec<f32>,        // [num_creative_plans * cx_dim]
    /// PFL3 99% contralateral bilinear move policy: Q_from, K_to, K_opp_def: [cx_dim * stream_b_dim].
    pub b_pfl3_q: Vec<f32>,
    pub b_pfl3_k: Vec<f32>,
    pub b_pfl3_opp: Vec<f32>,
    /// PFL2 50/50 bilateral sharpness/urgency gain: [stream_b_dim + cx_dim].
    pub b_pfl2_w: Vec<f32>,
    pub b_pfl2_b: f32,
    /// Deep WDL value head: [3 * (stream_b_dim + cx_dim)].
    pub b_wdl_w: Vec<f32>,
    pub b_wdl_b: [f32; 3],
    /// Strategic Pressure 10-20 ply forecast head: [5 * (stream_b_dim + cx_dim)].
    pub b_press_w: Vec<f32>,
    pub b_press_b: [f32; 5],
    /// Noduli Efference-Copy child eval delta predictor: [cx_dim] dot move_embed.
    pub b_noduli_w: Vec<f32>,      // [cx_dim]
    pub b_noduli_b: f32,
    /// Stream B -> Stream A Top-Down Seed & Centrifugal KC Mask low-rank projections:
    /// seed_proj_down: [seed_rank * (stream_b_dim + cx_dim)]
    /// seed_to_gamma:  [gamma_dim * seed_rank]
    /// seed_to_cent:   [acc_dim * seed_rank]
    pub b_seed_down: Vec<f32>,
    pub b_seed_to_gamma: Vec<f32>,
    pub b_seed_to_cent: Vec<f32>,

    /// B6. Lobula Dm/Pm Wide-Field 4-Head Retinotopic Spatial Interneurons (64x64 board vision):
    /// Links distant squares directly across ranks, files, diagonals, and knight leaps every tick!
    pub b_lob_q: Vec<f32>,         // [stream_b_dim * stream_b_dim]
    pub b_lob_k: Vec<f32>,         // [stream_b_dim * stream_b_dim]
    pub b_lob_v: Vec<f32>,         // [stream_b_dim * stream_b_dim]
    pub b_lob_o: Vec<f32>,         // [stream_b_dim * stream_b_dim]
    pub b_lob_rel: Vec<f32>,       // [4 * 64 * 64] learned relative chess-geometry bias
}

fn rand_uniform_scaled(rng: &mut u64, scale: f32) -> f32 {
    let u = (splitmix64(rng) >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
    ((u as f32) * 2.0 - 1.0) * scale
}

fn alloc_init(len: usize, scale: f32, rng: &mut u64) -> Vec<f32> {
    (0..len).map(|_| rand_uniform_scaled(rng, scale)).collect()
}

impl SaccadionWeights {
    /// Deterministic He/Xavier initialization matching our exact biological sign constraints.
    pub fn init_random(cfg: ModelConfig, seed: u64) -> Self {
        let mut rng = seed;
        let flyhash_claws = build_flyhash_claws(&cfg);
        let (_, gamma_dim, _) = cfg.lobe_dims_per_hemi();
        let plastic_dim = cfg.plastic_dim_per_hemi();
        let d = cfg.stream_b_dim;
        let cx = cfg.cx_dim;
        let mbon = cfg.mbon_dim;
        let stacks = cfg.num_phase_stacks;

        let s_ft = (1.0 / 32.0f32).sqrt() * 0.15;
        let s_d = (1.0 / d as f32).sqrt() * 0.5;
        let s_cx = (1.0 / cx as f32).sqrt() * 0.5;

        let a_ft_weight = alloc_init(cfg.sparse_input_dim * cfg.acc_dim, s_ft, &mut rng);
        let a_ft_bias = vec![0.05; cfg.acc_dim];
        let a_claw_weight = alloc_init(plastic_dim * cfg.flyhash_claws, 0.35, &mut rng);
        let a_apl_thresh = vec![0.18; cfg.acc_dim];

        let a_mbon_ipsi = alloc_init(stacks * cfg.acc_dim * mbon, (1.0 / (cfg.acc_dim as f32 * 0.1)).sqrt() * 0.25, &mut rng);
        let a_mbon_contra = alloc_init(stacks * cfg.acc_dim * mbon, (1.0 / (cfg.acc_dim as f32 * 0.1)).sqrt() * 0.25, &mut rng);
        let a_mbon_bias = vec![0.0; stacks * mbon];

        let a_lh_w_inh = alloc_init(cfg.lh_hidden_dim * cfg.lh_input_dim, 0.15, &mut rng);
        let a_lh_b_inh = vec![0.5; cfg.lh_hidden_dim];
        let a_lh_w_out = alloc_init(4 * cfg.lh_hidden_dim, 0.2, &mut rng);

        let smp_in = 2 * mbon + 4;
        let a_smp_w = alloc_init(stacks * 3 * smp_in, (1.0 / smp_in as f32).sqrt() * 0.3, &mut rng);
        let a_smp_b = vec![0.0; stacks * 3];

        Self {
            flyhash_claws,
            a_ft_weight,
            a_ft_bias,
            a_claw_weight,
            a_apl_beta: [0.85, 0.85],
            a_apl_thresh,
            a_mbon_ipsi,
            a_mbon_contra,
            a_mbon_bias,
            a_lh_w_inh,
            a_lh_b_inh,
            a_lh_w_out,
            a_lh_b_out: [-1.2, -1.5, -1.5, -1.2],
            a_smp_w,
            a_smp_b,

            b_shunt_exc: alloc_init(d * cfg.sq_feat_dim, s_d, &mut rng),
            b_shunt_inh: alloc_init(d * cfg.sq_feat_dim, s_d, &mut rng),
            b_shunt_leak: vec![0.6; d],
            b_lc_w: alloc_init(d * cfg.sq_feat_dim, s_d, &mut rng),
            b_optic_b: vec![0.0; d],

            b_hub_init: alloc_init((cfg.num_integrators + cfg.num_broadcasters) * d, s_d, &mut rng),
            b_int_q: alloc_init(d * d, s_d, &mut rng),
            b_int_k: alloc_init(d * d, s_d, &mut rng),
            b_int_v: alloc_init(d * d, s_d, &mut rng),
            b_int_o: alloc_init(d * d, s_d, &mut rng),
            b_rc_mix1: alloc_init(d * d, s_d, &mut rng),
            b_rc_mix2: alloc_init(d * d, s_d, &mut rng),
            b_rc_ln_g: vec![1.0; d],
            b_rc_ln_b: vec![0.0; d],

            b_pfn_stm: alloc_init(cfg.cx_sectors * cx * d, s_d, &mut rng),
            b_pfn_opp: alloc_init(cfg.cx_sectors * cx * d, s_d, &mut rng),
            b_vdelta: alloc_init(cx * cx, s_cx, &mut rng),
            b_hdelta: alloc_init(cx * cx, s_cx, &mut rng),
            b_delta7: alloc_init(cx * cx, s_cx, &mut rng),
            b_ring_inh: alloc_init(cx * (d + mbon), s_d, &mut rng),
            b_ring_b: vec![0.0; cx],

            b_cent_w: alloc_init(64 * d, s_d, &mut rng),
            b_cent_b: vec![0.5; 64],
            b_broad_to_sq: alloc_init(d * d, s_d, &mut rng),
            b_sq_upd: alloc_init(d * d, s_d, &mut rng),
            b_sq_ln_g: vec![1.0; d],
            b_sq_ln_b: vec![0.0; d],

            b_plan_bank: alloc_init(cfg.num_creative_plans * cx * (cx + d), s_d, &mut rng),
            b_plan_b: vec![0.0; cfg.num_creative_plans * cx],
            b_pfl3_q: alloc_init(cx * d, s_d, &mut rng),
            b_pfl3_k: alloc_init(cx * d, s_d, &mut rng),
            b_pfl3_opp: alloc_init(cx * d, s_d, &mut rng),
            b_pfl2_w: alloc_init(d + cx, s_d, &mut rng),
            b_pfl2_b: 0.0,
            b_wdl_w: alloc_init(3 * (d + cx), s_d, &mut rng),
            b_wdl_b: [0.0; 3],
            b_press_w: alloc_init(5 * (d + cx), s_d, &mut rng),
            b_press_b: [0.0; 5],
            b_noduli_w: alloc_init(cx, s_cx, &mut rng),
            b_noduli_b: 0.0,
            b_seed_down: alloc_init(cfg.seed_rank * (d + cx), s_d, &mut rng),
            b_seed_to_gamma: alloc_init(gamma_dim * cfg.seed_rank, 0.1, &mut rng),
            b_seed_to_cent: alloc_init(cfg.acc_dim * cfg.seed_rank, 0.1, &mut rng),
            b_lob_q: alloc_init(d * d, s_d, &mut rng),
            b_lob_k: alloc_init(d * d, s_d, &mut rng),
            b_lob_v: alloc_init(d * d, s_d, &mut rng),
            b_lob_o: alloc_init(d * d, s_d * 0.5, &mut rng),
            b_lob_rel: vec![0.0; 4 * 64 * 64],
            cfg,
        }
    }

    /// Total static parameter count (sparse_params, dense_params).
    pub fn param_counts(&self) -> (usize, usize) {
        let sparse = self.a_ft_weight.len() + self.a_ft_bias.len();
        let dense = self.a_claw_weight.len()
            + 2
            + self.a_apl_thresh.len()
            + self.a_mbon_ipsi.len()
            + self.a_mbon_contra.len()
            + self.a_mbon_bias.len()
            + self.a_lh_w_inh.len()
            + self.a_lh_b_inh.len()
            + self.a_lh_w_out.len()
            + 4
            + self.a_smp_w.len()
            + self.a_smp_b.len()
            + self.b_shunt_exc.len()
            + self.b_shunt_inh.len()
            + self.b_shunt_leak.len()
            + self.b_lc_w.len()
            + self.b_optic_b.len()
            + self.b_hub_init.len()
            + self.b_int_q.len()
            + self.b_int_k.len()
            + self.b_int_v.len()
            + self.b_int_o.len()
            + self.b_rc_mix1.len()
            + self.b_rc_mix2.len()
            + self.b_rc_ln_g.len()
            + self.b_rc_ln_b.len()
            + self.b_pfn_stm.len()
            + self.b_pfn_opp.len()
            + self.b_vdelta.len()
            + self.b_hdelta.len()
            + self.b_delta7.len()
            + self.b_ring_inh.len()
            + self.b_ring_b.len()
            + self.b_cent_w.len()
            + self.b_cent_b.len()
            + self.b_broad_to_sq.len()
            + self.b_sq_upd.len()
            + self.b_sq_ln_g.len()
            + self.b_sq_ln_b.len()
            + self.b_plan_bank.len()
            + self.b_plan_b.len()
            + self.b_pfl3_q.len()
            + self.b_pfl3_k.len()
            + self.b_pfl3_opp.len()
            + self.b_pfl2_w.len()
            + 1
            + self.b_wdl_w.len()
            + 3
            + self.b_press_w.len()
            + 5
            + self.b_noduli_w.len()
            + 1
            + self.b_seed_down.len()
            + self.b_seed_to_gamma.len()
            + self.b_seed_to_cent.len()
            + self.b_lob_q.len()
            + self.b_lob_k.len()
            + self.b_lob_v.len()
            + self.b_lob_o.len()
            + self.b_lob_rel.len();
        (sparse, dense)
    }

    /// Saves complete weights into canonical `.saccnet` binary format (little-endian FP32 or quantized).
    pub fn save_saccnet(&self, path: &Path) -> std::io::Result<()> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);

        // 80-byte deterministic header matching Python export_rust.py
        w.write_all(&SACCNET_MAGIC)?;
        w.write_all(&SACCNET_VERSION.to_le_bytes())?;
        w.write_all(&self.cfg.tier_id.to_le_bytes())?;
        for &val in &[
            self.cfg.sparse_input_dim as u32,
            self.cfg.acc_dim as u32,
            self.cfg.flyhash_claws as u32,
            self.cfg.mbon_dim as u32,
            self.cfg.num_phase_stacks as u32,
            self.cfg.lh_input_dim as u32,
            self.cfg.lh_hidden_dim as u32,
            self.cfg.sq_feat_dim as u32,
            self.cfg.stream_b_dim as u32,
            self.cfg.num_integrators as u32,
            self.cfg.num_broadcasters as u32,
            self.cfg.cx_sectors as u32,
            self.cfg.cx_dim as u32,
            self.cfg.cx_max_ticks as u32,
            self.cfg.num_creative_plans as u32,
            self.cfg.seed_rank as u32,
        ] {
            w.write_all(&val.to_le_bytes())?;
        }
        w.write_all(&self.cfg.flyhash_seed.to_le_bytes())?;

        let write_slice = |w: &mut BufWriter<File>, s: &[f32]| -> std::io::Result<()> {
            let bytes = unsafe {
                std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4)
            };
            w.write_all(bytes)
        };

        write_slice(&mut w, &self.a_ft_weight)?;
        write_slice(&mut w, &self.a_ft_bias)?;
        write_slice(&mut w, &self.a_claw_weight)?;
        write_slice(&mut w, &self.a_apl_beta)?;
        write_slice(&mut w, &self.a_apl_thresh)?;
        write_slice(&mut w, &self.a_mbon_ipsi)?;
        write_slice(&mut w, &self.a_mbon_contra)?;
        write_slice(&mut w, &self.a_mbon_bias)?;
        write_slice(&mut w, &self.a_lh_w_inh)?;
        write_slice(&mut w, &self.a_lh_b_inh)?;
        write_slice(&mut w, &self.a_lh_w_out)?;
        write_slice(&mut w, &self.a_lh_b_out)?;
        write_slice(&mut w, &self.a_smp_w)?;
        write_slice(&mut w, &self.a_smp_b)?;

        write_slice(&mut w, &self.b_shunt_exc)?;
        write_slice(&mut w, &self.b_shunt_inh)?;
        write_slice(&mut w, &self.b_shunt_leak)?;
        write_slice(&mut w, &self.b_lc_w)?;
        write_slice(&mut w, &self.b_optic_b)?;
        write_slice(&mut w, &self.b_hub_init)?;
        write_slice(&mut w, &self.b_int_q)?;
        write_slice(&mut w, &self.b_int_k)?;
        write_slice(&mut w, &self.b_int_v)?;
        write_slice(&mut w, &self.b_int_o)?;
        write_slice(&mut w, &self.b_rc_mix1)?;
        write_slice(&mut w, &self.b_rc_mix2)?;
        write_slice(&mut w, &self.b_rc_ln_g)?;
        write_slice(&mut w, &self.b_rc_ln_b)?;
        write_slice(&mut w, &self.b_pfn_stm)?;
        write_slice(&mut w, &self.b_pfn_opp)?;
        write_slice(&mut w, &self.b_vdelta)?;
        write_slice(&mut w, &self.b_hdelta)?;
        write_slice(&mut w, &self.b_delta7)?;
        write_slice(&mut w, &self.b_ring_inh)?;
        write_slice(&mut w, &self.b_ring_b)?;
        write_slice(&mut w, &self.b_cent_w)?;
        write_slice(&mut w, &self.b_cent_b)?;
        write_slice(&mut w, &self.b_broad_to_sq)?;
        write_slice(&mut w, &self.b_sq_upd)?;
        write_slice(&mut w, &self.b_sq_ln_g)?;
        write_slice(&mut w, &self.b_sq_ln_b)?;
        write_slice(&mut w, &self.b_plan_bank)?;
        write_slice(&mut w, &self.b_plan_b)?;
        write_slice(&mut w, &self.b_pfl3_q)?;
        write_slice(&mut w, &self.b_pfl3_k)?;
        write_slice(&mut w, &self.b_pfl3_opp)?;
        write_slice(&mut w, &self.b_pfl2_w)?;
        write_slice(&mut w, &[self.b_pfl2_b])?;
        write_slice(&mut w, &self.b_wdl_w)?;
        write_slice(&mut w, &self.b_wdl_b)?;
        write_slice(&mut w, &self.b_press_w)?;
        write_slice(&mut w, &self.b_press_b)?;
        write_slice(&mut w, &self.b_noduli_w)?;
        write_slice(&mut w, &[self.b_noduli_b])?;
        write_slice(&mut w, &self.b_seed_down)?;
        write_slice(&mut w, &self.b_seed_to_gamma)?;
        write_slice(&mut w, &self.b_seed_to_cent)?;
        write_slice(&mut w, &self.b_lob_q)?;
        write_slice(&mut w, &self.b_lob_k)?;
        write_slice(&mut w, &self.b_lob_v)?;
        write_slice(&mut w, &self.b_lob_o)?;
        write_slice(&mut w, &self.b_lob_rel)?;

        w.flush()
    }

    /// Loads a `.saccnet` binary weight file exported from JAX TPU training or Rust.
    pub fn load_saccnet(path: &Path) -> std::io::Result<Self> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);

        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if magic != SACCNET_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid SACCNET magic header",
            ));
        }

        let read_u32 = |r: &mut BufReader<File>| -> std::io::Result<u32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        let read_u64 = |r: &mut BufReader<File>| -> std::io::Result<u64> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b))
        };

        let ver = read_u32(&mut r)?;
        if ver != SACCNET_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported SACCNET version: {}", ver),
            ));
        }
        let tier_id = read_u32(&mut r)?;
        let cfg = ModelConfig {
            tier_id,
            sparse_input_dim: read_u32(&mut r)? as usize,
            acc_dim: read_u32(&mut r)? as usize,
            flyhash_claws: read_u32(&mut r)? as usize,
            mbon_dim: read_u32(&mut r)? as usize,
            num_phase_stacks: read_u32(&mut r)? as usize,
            lh_input_dim: read_u32(&mut r)? as usize,
            lh_hidden_dim: read_u32(&mut r)? as usize,
            sq_feat_dim: read_u32(&mut r)? as usize,
            stream_b_dim: read_u32(&mut r)? as usize,
            num_integrators: read_u32(&mut r)? as usize,
            num_broadcasters: read_u32(&mut r)? as usize,
            cx_sectors: read_u32(&mut r)? as usize,
            cx_dim: read_u32(&mut r)? as usize,
            cx_max_ticks: read_u32(&mut r)? as usize,
            num_creative_plans: read_u32(&mut r)? as usize,
            seed_rank: read_u32(&mut r)? as usize,
            flyhash_seed: read_u64(&mut r)?,
        };

        let read_vec = |r: &mut BufReader<File>, len: usize| -> std::io::Result<Vec<f32>> {
            let mut v = vec![0.0f32; len];
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len * 4)
            };
            r.read_exact(bytes)?;
            Ok(v)
        };

        let flyhash_claws = build_flyhash_claws(&cfg);
        let (_, gamma_dim, _) = cfg.lobe_dims_per_hemi();
        let plastic_dim = cfg.plastic_dim_per_hemi();
        let d = cfg.stream_b_dim;
        let cx = cfg.cx_dim;
        let mbon = cfg.mbon_dim;
        let stacks = cfg.num_phase_stacks;

        let a_ft_weight = read_vec(&mut r, cfg.sparse_input_dim * cfg.acc_dim)?;
        let a_ft_bias = read_vec(&mut r, cfg.acc_dim)?;
        let a_claw_weight = read_vec(&mut r, plastic_dim * cfg.flyhash_claws)?;
        let apl_v = read_vec(&mut r, 2)?;
        let a_apl_thresh = read_vec(&mut r, cfg.acc_dim)?;
        let a_mbon_ipsi = read_vec(&mut r, stacks * cfg.acc_dim * mbon)?;
        let a_mbon_contra = read_vec(&mut r, stacks * cfg.acc_dim * mbon)?;
        let a_mbon_bias = read_vec(&mut r, stacks * mbon)?;
        let a_lh_w_inh = read_vec(&mut r, cfg.lh_hidden_dim * cfg.lh_input_dim)?;
        let a_lh_b_inh = read_vec(&mut r, cfg.lh_hidden_dim)?;
        let a_lh_w_out = read_vec(&mut r, 4 * cfg.lh_hidden_dim)?;
        let lh_bo = read_vec(&mut r, 4)?;
        let a_smp_w = read_vec(&mut r, stacks * 3 * (2 * mbon + 4))?;
        let a_smp_b = read_vec(&mut r, stacks * 3)?;

        let b_shunt_exc = read_vec(&mut r, d * cfg.sq_feat_dim)?;
        let b_shunt_inh = read_vec(&mut r, d * cfg.sq_feat_dim)?;
        let b_shunt_leak = read_vec(&mut r, d)?;
        let b_lc_w = read_vec(&mut r, d * cfg.sq_feat_dim)?;
        let b_optic_b = read_vec(&mut r, d)?;
        let b_hub_init = read_vec(&mut r, (cfg.num_integrators + cfg.num_broadcasters) * d)?;
        let b_int_q = read_vec(&mut r, d * d)?;
        let b_int_k = read_vec(&mut r, d * d)?;
        let b_int_v = read_vec(&mut r, d * d)?;
        let b_int_o = read_vec(&mut r, d * d)?;
        let b_rc_mix1 = read_vec(&mut r, d * d)?;
        let b_rc_mix2 = read_vec(&mut r, d * d)?;
        let b_rc_ln_g = read_vec(&mut r, d)?;
        let b_rc_ln_b = read_vec(&mut r, d)?;
        let b_pfn_stm = read_vec(&mut r, cfg.cx_sectors * cx * d)?;
        let b_pfn_opp = read_vec(&mut r, cfg.cx_sectors * cx * d)?;
        let b_vdelta = read_vec(&mut r, cx * cx)?;
        let b_hdelta = read_vec(&mut r, cx * cx)?;
        let b_delta7 = read_vec(&mut r, cx * cx)?;
        let b_ring_inh = read_vec(&mut r, cx * (d + mbon))?;
        let b_ring_b = read_vec(&mut r, cx)?;
        let b_cent_w = read_vec(&mut r, 64 * d)?;
        let b_cent_b = read_vec(&mut r, 64)?;
        let b_broad_to_sq = read_vec(&mut r, d * d)?;
        let b_sq_upd = read_vec(&mut r, d * d)?;
        let b_sq_ln_g = read_vec(&mut r, d)?;
        let b_sq_ln_b = read_vec(&mut r, d)?;
        let b_plan_bank = read_vec(&mut r, cfg.num_creative_plans * cx * (cx + d))?;
        let b_plan_b = read_vec(&mut r, cfg.num_creative_plans * cx)?;
        let b_pfl3_q = read_vec(&mut r, cx * d)?;
        let b_pfl3_k = read_vec(&mut r, cx * d)?;
        let b_pfl3_opp = read_vec(&mut r, cx * d)?;
        let b_pfl2_w = read_vec(&mut r, d + cx)?;
        let b_pfl2_b = read_vec(&mut r, 1)?[0];
        let b_wdl_w = read_vec(&mut r, 3 * (d + cx))?;
        let wdl_b = read_vec(&mut r, 3)?;
        let b_press_w = read_vec(&mut r, 5 * (d + cx))?;
        let press_b = read_vec(&mut r, 5)?;
        let b_noduli_w = read_vec(&mut r, cx)?;
        let b_noduli_b = read_vec(&mut r, 1)?[0];
        let b_seed_down = read_vec(&mut r, cfg.seed_rank * (d + cx))?;
        let b_seed_to_gamma = read_vec(&mut r, gamma_dim * cfg.seed_rank)?;
        let b_seed_to_cent = read_vec(&mut r, cfg.acc_dim * cfg.seed_rank)?;
        let b_lob_q = read_vec(&mut r, d * d)?;
        let b_lob_k = read_vec(&mut r, d * d)?;
        let b_lob_v = read_vec(&mut r, d * d)?;
        let b_lob_o = read_vec(&mut r, d * d)?;
        let b_lob_rel = read_vec(&mut r, 4 * 64 * 64)?;

        Ok(Self {
            cfg,
            flyhash_claws,
            a_ft_weight,
            a_ft_bias,
            a_claw_weight,
            a_apl_beta: [apl_v[0], apl_v[1]],
            a_apl_thresh,
            a_mbon_ipsi,
            a_mbon_contra,
            a_mbon_bias,
            a_lh_w_inh,
            a_lh_b_inh,
            a_lh_w_out,
            a_lh_b_out: [lh_bo[0], lh_bo[1], lh_bo[2], lh_bo[3]],
            a_smp_w,
            a_smp_b,
            b_shunt_exc,
            b_shunt_inh,
            b_shunt_leak,
            b_lc_w,
            b_optic_b,
            b_hub_init,
            b_int_q,
            b_int_k,
            b_int_v,
            b_int_o,
            b_rc_mix1,
            b_rc_mix2,
            b_rc_ln_g,
            b_rc_ln_b,
            b_pfn_stm,
            b_pfn_opp,
            b_vdelta,
            b_hdelta,
            b_delta7,
            b_ring_inh,
            b_ring_b,
            b_cent_w,
            b_cent_b,
            b_broad_to_sq,
            b_sq_upd,
            b_sq_ln_g,
            b_sq_ln_b,
            b_plan_bank,
            b_plan_b,
            b_pfl3_q,
            b_pfl3_k,
            b_pfl3_opp,
            b_pfl2_w,
            b_pfl2_b,
            b_wdl_w,
            b_wdl_b: [wdl_b[0], wdl_b[1], wdl_b[2]],
            b_press_w,
            b_press_b: [press_b[0], press_b[1], press_b[2], press_b[3], press_b[4]],
            b_noduli_w,
            b_noduli_b,
            b_seed_down,
            b_seed_to_gamma,
            b_seed_to_cent,
            b_lob_q,
            b_lob_k,
            b_lob_v,
            b_lob_o,
            b_lob_rel,
        })
    }
}
