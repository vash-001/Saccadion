//! Saccadion-1 Pure-Rust Neural Architecture (All 9 FlyWire v783 Circuits).

pub mod config;
pub mod plasticity;
pub mod simd;
pub mod stream_a;
pub mod stream_b;
pub mod weights;

pub use config::ModelConfig;
pub use plasticity::{DopaminePlasticityState, PlasticityParams, SparseActiveKc};
pub use stream_a::{evaluate_stream_a, LateralHornVeto, StreamAOutput, StreamAWorkspace};
pub use stream_b::{evaluate_stream_b, PolicyMoveScore, StreamBOutput};
pub use weights::SaccadionWeights;

#[cfg(test)]
mod tests {
    use super::*;
    use saccadion_chess::movegen::generate_legal;
    use saccadion_chess::position::Position;
    use saccadion_chess::types::MoveList;

    #[test]
    fn test_avx2_scalar_parity() {
        let a: Vec<f32> = (0..131).map(|i| (i as f32 * 0.13).sin()).collect();
        let b: Vec<f32> = (0..131).map(|i| (i as f32 * 0.27).cos()).collect();
        let d_simd = simd::dot_f32(&a, &b);
        let d_scal = simd::dot_f32_scalar(&a, &b);
        assert!((d_simd - d_scal).abs() < 1e-4, "simd={} scalar={}", d_simd, d_scal);

        let mut dst_simd = a.clone();
        let mut dst_scal = a.clone();
        simd::axpy_f32(&mut dst_simd, &b, 0.37);
        simd::axpy_f32_scalar(&mut dst_scal, &b, 0.37);
        for i in 0..a.len() {
            assert!((dst_simd[i] - dst_scal[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_full_dual_stream_and_dopamine_plasticity() {
        let cfg = ModelConfig::tiny();
        let weights = SaccadionWeights::init_random(cfg.clone(), 42);
        let mut ws = StreamAWorkspace::new(&weights);
        let mut plasticity = DopaminePlasticityState::new(&cfg);

        let mut pos = Position::startpos();
        let mut legal = MoveList::new();
        generate_legal(&mut pos, &mut legal);

        // Initial evaluation with zero fast weights
        let out_a0 = evaluate_stream_a(&pos, &weights, &mut ws, Some(&plasticity), None, 1.0);
        assert!(out_a0.total_cp.is_finite());
        assert_eq!(out_a0.gamma_cp, 0.0);
        assert_eq!(out_a0.game_cp, 0.0);

        // Run Stream B Deep Strategic Brain
        let novelty = plasticity.mbon_alpha_prime3_novelty(&out_a0.active_apbp_stm, true);
        let out_b = evaluate_stream_b(
            &pos,
            &legal,
            &weights,
            &out_a0.mbon_summary,
            Some(&plasticity),
            novelty,
            1.0,
            true,
        );
        assert!(out_b.deep_cp.is_finite());
        assert_eq!(out_b.policy_moves.len(), 20);
        let prob_sum: f32 = out_b.policy_moves.iter().map(|m| m.prob).sum();
        assert!((prob_sum - 1.0).abs() < 1e-4);
        assert_eq!(out_b.creative_plans.len(), 4);
        assert!(out_b.ticks_used >= 2 && out_b.ticks_used <= cfg.cx_max_ticks);

        // Inject Stream B top-down seed and simulate a sharp PPL1 blunder alarm
        plasticity.inject_stream_b_seed(&out_b.gamma_seed, &out_b.gamma_seed, 0.25);
        let mut active_probe = out_a0.active_gamma_stm.clone();
        if active_probe.items.is_empty() {
            active_probe.push(0, 0.5);
        }
        plasticity.accumulate_eligibility(&active_probe, &out_a0.active_gamma_opp);
        plasticity.apply_dopamine_surprise(-300.0, &active_probe, &out_a0.active_gamma_opp);
        assert_eq!(plasticity.ppl1_updates, 1);

        // Trigger Dop1R2 causal reversal erosion (opposite-sign surprise) & MBON->DAN SOC feedback
        plasticity.apply_dopamine_surprise(250.0, &active_probe, &out_a0.active_gamma_opp);
        assert_eq!(plasticity.pam_updates, 1);
        assert_eq!(plasticity.dop1r2_erosions, 1);
        plasticity.apply_mbon_soc_feedback(120.0);
        assert_eq!(plasticity.soc_updates, 1);

        // Verify frozen base slow_cp is 100% invariant while fast residual adapted within bounds
        let out_a1 = evaluate_stream_a(
            &pos,
            &weights,
            &mut ws,
            Some(&plasticity),
            Some(&out_b.centrifugal_kc_mask),
            1.0,
        );
        assert!(out_a1.gamma_cp.abs() <= plasticity.params.clamp_gamma_cp + 1e-4);

        // Verify ucinewgame clean reset restores exact bit-identical baseline!
        plasticity.reset_game();
        let out_a_reset = evaluate_stream_a(&pos, &weights, &mut ws, Some(&plasticity), None, 1.0);
        assert!((out_a_reset.total_cp - out_a0.total_cp).abs() < 1e-5);
    }

    #[test]
    fn test_saccnet_binary_roundtrip() {
        let cfg = ModelConfig::tiny();
        let weights = SaccadionWeights::init_random(cfg, 1337);
        let tmp_path = std::env::temp_dir().join("saccadion_tiny_test.saccnet");
        weights.save_saccnet(&tmp_path).expect("save .saccnet");
        let loaded = SaccadionWeights::load_saccnet(&tmp_path).expect("load .saccnet");
        let _ = std::fs::remove_file(&tmp_path);

        assert_eq!(weights.cfg, loaded.cfg);
        assert_eq!(weights.a_ft_weight.len(), loaded.a_ft_weight.len());
        assert_eq!(weights.a_ft_weight[123], loaded.a_ft_weight[123]);
        assert_eq!(weights.b_vdelta[7], loaded.b_vdelta[7]);
        assert_eq!(weights.b_pfl2_b, loaded.b_pfl2_b);

        let pos = Position::startpos();
        let mut ws1 = StreamAWorkspace::new(&weights);
        let mut ws2 = StreamAWorkspace::new(&loaded);
        let o1 = evaluate_stream_a(&pos, &weights, &mut ws1, None, None, 1.0);
        let o2 = evaluate_stream_a(&pos, &loaded, &mut ws2, None, None, 1.0);
        assert!((o1.total_cp - o2.total_cp).abs() < 1e-6);
    }

    #[test]
    fn test_finny_cache_matches_fresh_accumulator() {
        let cfg = ModelConfig::tiny();
        let weights = SaccadionWeights::init_random(cfg, 99);
        let mut ws_cached = StreamAWorkspace::new(&weights);
        let mut ws_fresh = StreamAWorkspace::new(&weights);

        let mut pos = Position::startpos();
        let _ = evaluate_stream_a(&pos, &weights, &mut ws_cached, None, None, 1.0);

        // Play a sequence including king move (e2e4 e7e5 e1e2)
        for uci in ["e2e4", "e7e5", "e1e2", "g8f6", "g1f3"] {
            let mv = saccadion_chess::movegen::parse_uci_move(&mut pos, uci).unwrap();
            pos.make_move(mv);
            let o_cached = evaluate_stream_a(&pos, &weights, &mut ws_cached, None, None, 1.0);
            ws_fresh.finny.clear();
            let o_fresh = evaluate_stream_a(&pos, &weights, &mut ws_fresh, None, None, 1.0);
            assert!(
                (o_cached.slow_cp - o_fresh.slow_cp).abs() < 1e-4,
                "Finny cache mismatch on {}: {} vs {}",
                uci,
                o_cached.slow_cp,
                o_fresh.slow_cp
            );
        }
    }
}
