//! Frozen-Yet-Adaptive Mushroom Body Fast Weights, Eligibility Traces, MBON-a'3 Novelty,
//! and LAL Bistable Commit State.
//!
//! All base weights W_slow remain 100% frozen at inference. Only the small L1-resident
//! gamma (intra-search) and alpha'/beta' (intra-game) plasticity vectors adapt in < 50 ns
//! via asymmetric PAM (opportunity) and 12x denser bilateral PPL1 (blunder alarm) Dopamine.

use crate::config::ModelConfig;

/// Hyperparameters controlling three-factor causal eligibility-trace plasticity and bounds.
///
/// Implements peer-reviewed Drosophila causal learning circuits:
/// - Dop1R1 (G_alpha_s / cAMP): forward causal credit along eligibility trace (`CS -> US`)
/// - Dop1R2 (G_alpha_q / IP3): temporal causal order erosion of non-causal correlations / reversals
/// - MBON -> DAN Second-Order Conditioning (SOC): stable alpha/beta MBON shifts generate
///   internal pseudo-dopamine to train fast gamma weights before delayed primary reward arrives.
#[derive(Clone, Debug)]
pub struct PlasticityParams {
    pub eta_pam: f32,          // Positive strategic opportunity learning rate (Dop1R1)
    pub eta_ppl1: f32,         // Sharp negative blunder-alarm learning rate (~3.5x PAM)
    pub eta_dop1r2: f32,       // Dop1R2 causal order reversal / active forgetting rate
    pub eta_soc: f32,          // MBON -> DAN Second-Order Conditioning coupling strength
    pub ppl1_contra_ratio: f32,// FlyWire v783 measured contralateral PPL1 ratio (0.332)
    pub lambda_gamma: f32,     // Per-update exponential decay on intra-search fast weights
    pub lambda_game: f32,      // Per-move exponential decay on intra-game fast weights (DPM)
    pub trace_decay: f32,      // Biochemical cAMP/Ca2+ eligibility trace decay rho across plies
    pub clamp_gamma_cp: f32,   // Hard minimax/TT stability bound on gamma residual (+/- cp)
    pub clamp_game_cp: f32,    // Hard minimax/TT stability bound on game residual (+/- cp)
    pub surprise_thresh_cp: f32,// Minimum |V_search - V_net| to fire Dopamine update
    pub dpm_consolidate_lr: f32,// Consolidation rate from gamma root lesson into alpha'/beta'
}

impl Default for PlasticityParams {
    fn default() -> Self {
        Self {
            eta_pam: 0.08,
            eta_ppl1: 0.28,
            eta_dop1r2: 0.14,
            eta_soc: 0.22,
            ppl1_contra_ratio: 0.332,
            lambda_gamma: 0.02,
            lambda_game: 0.08,
            trace_decay: 0.65,
            clamp_gamma_cp: 120.0,
            clamp_game_cp: 80.0,
            surprise_thresh_cp: 35.0,
            dpm_consolidate_lr: 0.15,
        }
    }
}

/// Sparse active list of (index, activation) pairs produced by APL 92%-sparse inhibition.
#[derive(Clone, Debug, Default)]
pub struct SparseActiveKc {
    pub items: Vec<(u16, f32)>,
}

impl SparseActiveKc {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            items: Vec::with_capacity(cap),
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    #[inline(always)]
    pub fn push(&mut self, idx: u16, val: f32) {
        self.items.push((idx, val));
    }

    pub fn l2_sq(&self) -> f32 {
        self.items.iter().map(|&(_, v)| v * v).sum()
    }
}

/// Per-thread L1-resident plasticity, eligibility trace, novelty suppression, and LAL commit state.
#[derive(Clone, Debug)]
pub struct DopaminePlasticityState {
    pub params: PlasticityParams,
    pub gamma_dim: usize,
    pub apbp_dim: usize,
    /// Intra-search fast weights per hemisphere (reset or decayed per search).
    pub dw_gamma_stm: Vec<f32>,
    pub dw_gamma_opp: Vec<f32>,
    /// Intra-game fast weights per hemisphere (persists across moves, zeroed on ucinewgame).
    pub dw_apbp_stm: Vec<f32>,
    pub dw_apbp_opp: Vec<f32>,
    /// Biochemical eligibility trace vectors accumulated along the current search stack.
    pub elig_gamma_stm: Vec<f32>,
    pub elig_gamma_opp: Vec<f32>,
    /// MBON-alpha'3 familiarity counts (indexed by hashed active KC signature) for novelty/creativity.
    pub familiarity_table: Vec<f32>,
    /// LAL bistable previous-move strategic goal vector (prevents move-to-move plan oscillation).
    pub lal_committed_goal: Vec<f32>,
    /// Previous surprise delta for Dop1R2 causal-reversal detection.
    pub prev_delta: f32,
    /// Previous frozen alpha/beta MBON valuation (cp) for MBON -> DAN second-order conditioning.
    pub prev_mbon_cp: f32,
    /// Total dopamine events fired during the current search (diagnostics).
    pub pam_updates: u64,
    pub ppl1_updates: u64,
    pub dop1r2_erosions: u64,
    pub soc_updates: u64,
}

impl DopaminePlasticityState {
    pub fn new(cfg: &ModelConfig) -> Self {
        let (_, gamma_dim, apbp_dim) = cfg.lobe_dims_per_hemi();
        Self {
            params: PlasticityParams::default(),
            gamma_dim,
            apbp_dim,
            dw_gamma_stm: vec![0.0; gamma_dim],
            dw_gamma_opp: vec![0.0; gamma_dim],
            dw_apbp_stm: vec![0.0; apbp_dim],
            dw_apbp_opp: vec![0.0; apbp_dim],
            elig_gamma_stm: vec![0.0; gamma_dim],
            elig_gamma_opp: vec![0.0; gamma_dim],
            familiarity_table: vec![0.0; 256],
            lal_committed_goal: vec![0.0; cfg.cx_dim],
            prev_delta: 0.0,
            prev_mbon_cp: 0.0,
            pam_updates: 0,
            ppl1_updates: 0,
            dop1r2_erosions: 0,
            soc_updates: 0,
        }
    }

    /// Cleanly resets ALL fast weights, traces, familiarity, and LAL commit (called on `ucinewgame`).
    pub fn reset_game(&mut self) {
        self.dw_gamma_stm.fill(0.0);
        self.dw_gamma_opp.fill(0.0);
        self.dw_apbp_stm.fill(0.0);
        self.dw_apbp_opp.fill(0.0);
        self.elig_gamma_stm.fill(0.0);
        self.elig_gamma_opp.fill(0.0);
        self.familiarity_table.fill(0.0);
        self.lal_committed_goal.fill(0.0);
        self.prev_delta = 0.0;
        self.prev_mbon_cp = 0.0;
        self.pam_updates = 0;
        self.ppl1_updates = 0;
        self.dop1r2_erosions = 0;
        self.soc_updates = 0;
    }

    /// Softly decays intra-search gamma weights and clears eligibility traces before a new root search.
    pub fn prepare_new_search(&mut self) {
        for x in &mut self.dw_gamma_stm {
            *x *= 0.25;
        }
        for x in &mut self.dw_gamma_opp {
            *x *= 0.25;
        }
        self.elig_gamma_stm.fill(0.0);
        self.elig_gamma_opp.fill(0.0);
        self.prev_delta = 0.0;
        self.pam_updates = 0;
        self.ppl1_updates = 0;
        self.dop1r2_erosions = 0;
        self.soc_updates = 0;
    }

    /// Injects a top-down strategic fast-weight seed from Stream B into intra-search gamma weights.
    pub fn inject_stream_b_seed(&mut self, seed_stm: &[f32], seed_opp: &[f32], strength: f32) {
        let n = self.gamma_dim.min(seed_stm.len()).min(seed_opp.len());
        for i in 0..n {
            self.dw_gamma_stm[i] = (self.dw_gamma_stm[i] * (1.0 - strength) + seed_stm[i] * strength)
                .clamp(-1.5, 1.5);
            self.dw_gamma_opp[i] = (self.dw_gamma_opp[i] * (1.0 - strength) + seed_opp[i] * strength)
                .clamp(-1.5, 1.5);
        }
    }

    /// Accumulates active sparse gamma Kenyon Cells onto the decaying eligibility trace:
    /// `e(t) = rho * e(t-1) + k_gamma(s_t)` (Handler et al., Cell 2019).
    pub fn accumulate_eligibility(&mut self, active_stm: &SparseActiveKc, active_opp: &SparseActiveKc) {
        let rho = self.params.trace_decay;
        for x in &mut self.elig_gamma_stm {
            *x *= rho;
        }
        for x in &mut self.elig_gamma_opp {
            *x *= rho;
        }
        for &(idx, val) in &active_stm.items {
            let i = idx as usize;
            if i < self.gamma_dim {
                self.elig_gamma_stm[i] += val;
            }
        }
        for &(idx, val) in &active_opp.items {
            let i = idx as usize;
            if i < self.gamma_dim {
                self.elig_gamma_opp[i] += val;
            }
        }
    }

    /// Evaluates the bounded fast-weight residual (in centipawns) for active sparse gamma & alpha'/beta' KCs.
    pub fn evaluate_residual_cp(
        &self,
        gamma_stm: &SparseActiveKc,
        gamma_opp: &SparseActiveKc,
        apbp_stm: &SparseActiveKc,
        apbp_opp: &SparseActiveKc,
    ) -> (f32, f32) {
        let mut raw_gamma = 0.0f32;
        for &(idx, val) in &gamma_stm.items {
            raw_gamma += self.dw_gamma_stm[idx as usize] * val;
        }
        for &(idx, val) in &gamma_opp.items {
            raw_gamma -= 0.5 * self.dw_gamma_opp[idx as usize] * val;
        }

        let mut raw_game = 0.0f32;
        for &(idx, val) in &apbp_stm.items {
            raw_game += self.dw_apbp_stm[idx as usize] * val;
        }
        for &(idx, val) in &apbp_opp.items {
            raw_game -= 0.5 * self.dw_apbp_opp[idx as usize] * val;
        }

        // Convert normalized fast-weight dot product into bounded centipawn residuals
        let gamma_cp = (raw_gamma * 100.0).clamp(-self.params.clamp_gamma_cp, self.params.clamp_gamma_cp);
        let game_cp = (raw_game * 100.0).clamp(-self.params.clamp_game_cp, self.params.clamp_game_cp);
        (gamma_cp, game_cp)
    }

    /// Applies an asymmetric three-factor PAM / PPL1 Dopamine update triggered by search surprise
    /// `delta_cp = V_search - V_static`.
    pub fn apply_dopamine_surprise(
        &mut self,
        delta_cp: f32,
        gamma_stm: &SparseActiveKc,
        gamma_opp: &SparseActiveKc,
    ) {
        if delta_cp.abs() < self.params.surprise_thresh_cp {
            return;
        }
        // Normalize surprise into [-2.0, +2.0] pawn units for numerical stability
        let delta = (delta_cp / 200.0).clamp(-2.0, 2.0);
        let decay = 1.0 - self.params.lambda_gamma;

        for x in &mut self.dw_gamma_stm {
            *x *= decay;
        }
        for x in &mut self.dw_gamma_opp {
            *x *= decay;
        }

        // Combine immediate active KCs with biochemical eligibility trace
        let mut stm_norm_sq = gamma_stm.l2_sq() + 1e-4;
        let mut opp_norm_sq = gamma_opp.l2_sq() + 1e-4;
        for &e in &self.elig_gamma_stm {
            stm_norm_sq += 0.25 * e * e;
        }
        for &e in &self.elig_gamma_opp {
            opp_norm_sq += 0.25 * e * e;
        }

        // Dop1R2 (G_alpha_q / IP3) bidirectional temporal causal order gate:
        // When surprise reverses sign (prev_delta * delta < 0), actively erode stale non-causal
        // fast weights on currently active KCs to eliminate horizon/trap illusions.
        if self.prev_delta * delta < -0.02 {
            self.dop1r2_erosions += 1;
            let erode = (1.0 - self.params.eta_dop1r2 * delta.abs().min(1.0)).clamp(0.4, 1.0);
            for &(idx, _) in &gamma_stm.items {
                self.dw_gamma_stm[idx as usize] *= erode;
            }
            for &(idx, _) in &gamma_opp.items {
                self.dw_gamma_opp[idx as usize] *= erode;
            }
        }
        self.prev_delta = delta;

        if delta > 0.0 {
            // Dop1R1 + PAM positive causal opportunity Dopamine
            self.pam_updates += 1;
            let step = (self.params.eta_pam * delta) / stm_norm_sq;
            for &(idx, val) in &gamma_stm.items {
                let i = idx as usize;
                self.dw_gamma_stm[i] = (self.dw_gamma_stm[i] + step * val).clamp(-2.0, 2.0);
            }
            // Retroactive forward-causal credit along eligibility trace
            let trace_step = 0.5 * step;
            for i in 0..self.gamma_dim {
                let e = self.elig_gamma_stm[i];
                if e.abs() > 1e-4 {
                    self.dw_gamma_stm[i] = (self.dw_gamma_stm[i] + trace_step * e).clamp(-2.0, 2.0);
                }
            }
        } else {
            // Dop1R1 + PPL1 sharp negative blunder alarm (12x denser per neuron, 33.2% contralateral!)
            self.ppl1_updates += 1;
            let mag = -delta;
            let step_stm = (self.params.eta_ppl1 * mag) / stm_norm_sq;
            let step_opp = (self.params.eta_ppl1 * self.params.ppl1_contra_ratio * mag) / opp_norm_sq;
            for &(idx, val) in &gamma_stm.items {
                let i = idx as usize;
                self.dw_gamma_stm[i] = (self.dw_gamma_stm[i] - step_stm * val).clamp(-2.0, 2.0);
            }
            for &(idx, val) in &gamma_opp.items {
                let i = idx as usize;
                self.dw_gamma_opp[i] = (self.dw_gamma_opp[i] + step_opp * val).clamp(-2.0, 2.0);
            }
            let trace_stm = 0.5 * step_stm;
            for i in 0..self.gamma_dim {
                let e = self.elig_gamma_stm[i];
                if e.abs() > 1e-4 {
                    self.dw_gamma_stm[i] = (self.dw_gamma_stm[i] - trace_stm * e).clamp(-2.0, 2.0);
                }
            }
        }
    }

    /// MBON -> DAN Second-Order Conditioning (SOC) feedback loop (Felsenberg 2018; Yamada 2023):
    /// When the stable frozen alpha/beta MBON valuation shifts cleanly across a strategic sequence
    /// `A -> B` before primary tactical reward arrives at `C`, it drives internal pseudo-dopamine
    /// onto the eligibility trace of the preceding cause `A`.
    pub fn apply_mbon_soc_feedback(&mut self, mbon_slow_cp: f32) {
        let diff_cp = mbon_slow_cp - self.prev_mbon_cp;
        self.prev_mbon_cp = mbon_slow_cp;
        if diff_cp.abs() < 25.0 {
            return;
        }
        let soc_delta = (diff_cp / 250.0).clamp(-1.0, 1.0) * self.params.eta_soc;
        let mut trace_norm_sq = 0.0f32;
        for &e in &self.elig_gamma_stm {
            trace_norm_sq += e * e;
        }
        if trace_norm_sq < 1e-8 {
            return;
        }
        self.soc_updates += 1;
        let step = soc_delta / (trace_norm_sq + 1e-4);
        for i in 0..self.gamma_dim {
            let e = self.elig_gamma_stm[i];
            if e.abs() > 1e-4 {
                self.dw_gamma_stm[i] = (self.dw_gamma_stm[i] + step * e).clamp(-2.0, 2.0);
            }
        }
    }

    /// DPM consolidation between played moves: transfers root surprise slowly into intra-game
    /// alpha'/beta' fast weights so the engine adapts to the game's strategic character.
    pub fn dpm_consolidate_move(
        &mut self,
        root_delta_cp: f32,
        apbp_stm: &SparseActiveKc,
        apbp_opp: &SparseActiveKc,
    ) {
        let keep = 1.0 - self.params.lambda_game;
        for x in &mut self.dw_apbp_stm {
            *x *= keep;
        }
        for x in &mut self.dw_apbp_opp {
            *x *= keep;
        }
        let delta = (root_delta_cp / 250.0).clamp(-1.5, 1.5);
        let norm_stm = apbp_stm.l2_sq() + 1e-4;
        let norm_opp = apbp_opp.l2_sq() + 1e-4;
        let lr = self.params.dpm_consolidate_lr;
        for &(idx, val) in &apbp_stm.items {
            let i = idx as usize;
            self.dw_apbp_stm[i] = (self.dw_apbp_stm[i] + (lr * delta * val) / norm_stm).clamp(-1.5, 1.5);
        }
        for &(idx, val) in &apbp_opp.items {
            let i = idx as usize;
            self.dw_apbp_opp[i] = (self.dw_apbp_opp[i] - (0.33 * lr * delta * val) / norm_opp).clamp(-1.5, 1.5);
        }
    }

    /// Computes MBON-alpha'3 novelty score in [0, 1] (high for novel patterns, suppressed for stale repeats)
    /// and updates repetition-suppression familiarity.
    pub fn mbon_alpha_prime3_novelty(&mut self, active_apbp: &SparseActiveKc, record: bool) -> f32 {
        let mut sig = 0x517C_C1B7_2722_0A95u64;
        for &(idx, _) in &active_apbp.items {
            sig = sig.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (idx as u64);
        }
        let bucket = (sig as usize) & 255;
        let fam = self.familiarity_table[bucket];
        let novelty = 1.0 / (1.0 + fam);
        if record {
            // Gentle decay of all buckets + increment visited pattern
            for f in &mut self.familiarity_table {
                *f *= 0.985;
            }
            self.familiarity_table[bucket] = (self.familiarity_table[bucket] + 1.0).min(8.0);
        }
        novelty
    }

    /// Updates the LAL bistable committed strategic goal vector with smooth exponential inertia.
    pub fn commit_lal_goal(&mut self, goal_vec: &[f32]) {
        if self.lal_committed_goal.len() != goal_vec.len() {
            self.lal_committed_goal = goal_vec.to_vec();
            return;
        }
        for i in 0..goal_vec.len() {
            self.lal_committed_goal[i] = 0.65 * self.lal_committed_goal[i] + 0.35 * goal_vec[i];
        }
    }

    /// Cosine alignment between a candidate goal vector and the LAL committed plan in [-1, 1].
    pub fn lal_alignment(&self, candidate_goal: &[f32]) -> f32 {
        let n = self.lal_committed_goal.len().min(candidate_goal.len());
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for i in 0..n {
            dot += self.lal_committed_goal[i] * candidate_goal[i];
            na += self.lal_committed_goal[i] * self.lal_committed_goal[i];
            nb += candidate_goal[i] * candidate_goal[i];
        }
        if na < 1e-6 || nb < 1e-6 {
            0.0
        } else {
            (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0)
        }
    }
}
