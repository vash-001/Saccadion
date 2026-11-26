//! Saccadion Neuro-Guided Hybrid CPU Search crate.

pub mod search;
pub mod tt;

pub use search::{SearchConfig, SearchEngine, SearchResult};
pub use tt::{StrategicTT, TranspositionTable, MATE_SCORE, MATE_THRESHOLD};

#[cfg(test)]
mod tests {
    use super::*;
    use saccadion_chess::position::Position;
    use saccadion_nn::config::ModelConfig;
    use saccadion_nn::weights::SaccadionWeights;

    #[test]
    fn test_mate_in_1_and_mate_in_2_tactical_solver() {
        let weights = SaccadionWeights::init_random(ModelConfig::tiny(), 77);
        let mut engine = SearchEngine::new(weights, 8);

        // Classic Mate-in-1: White Queen on g6/h5 mates on f7 (Qxf7#)
        let mut pos1 = Position::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4").unwrap();
        let cfg = SearchConfig {
            max_depth: 3,
            cx_min_depth: 2,
            ..SearchConfig::default()
        };
        let res1 = engine.search(&mut pos1, cfg.clone(), |_| {});
        assert_eq!(res1.best_move.to_uci(), "h5f7");
        assert!(res1.score_cp > MATE_THRESHOLD);
        assert!(res1.stream_b_evals >= 1);

        // Classic Mate-in-2: Back-rank sacrifice (1. Re8+ Rxe8 2. Rxe8#)
        engine.new_game();
        let mut pos2 = Position::from_fen("3r2k1/5ppp/8/8/8/8/5PPP/3RR1K1 w - - 0 1").unwrap();
        let cfg2 = SearchConfig {
            max_depth: 5,
            cx_min_depth: 3,
            ..SearchConfig::default()
        };
        let res2 = engine.search(&mut pos2, cfg2, |_| {});
        assert!(res2.score_cp > MATE_THRESHOLD, "expected forced mate, got {}", res2.score_cp);
        assert!(
            res2.best_move.to_uci() == "e1e8" || res2.best_move.to_uci() == "d1d8",
            "unexpected mate-in-2 move: {}",
            res2.best_move.to_uci()
        );
    }

    #[test]
    fn test_startpos_search_and_clean_ucinewgame_reset() {
        let weights = SaccadionWeights::init_random(ModelConfig::tiny(), 2026);
        let mut engine = SearchEngine::new(weights, 8);
        let mut pos = Position::startpos();
        let cfg = SearchConfig {
            max_depth: 4,
            cx_min_depth: 3,
            ..SearchConfig::default()
        };
        let res = engine.search(&mut pos, cfg, |_| {});
        assert!(!res.best_move.is_null());
        assert!(res.nodes > 50);
        assert!(res.stream_b_evals >= 1);

        // Verify ucinewgame resets all fast plastic weights back to exact zero
        engine.new_game();
        assert!(engine.plasticity.dw_gamma_stm.iter().all(|&x| x == 0.0));
        assert!(engine.plasticity.dw_apbp_stm.iter().all(|&x| x == 0.0));

        // Verify Parallel DNp Multi-Core Lazy SMP (threads=2) completes cleanly on shared lock-free TT
        let mut pos_mp = Position::startpos();
        let cfg_mp = SearchConfig {
            max_depth: 4,
            threads: 2,
            cx_min_depth: 4,
            ..SearchConfig::default()
        };
        let res_mp = engine.search(&mut pos_mp, cfg_mp, |_| {});
        assert!(!res_mp.best_move.is_null());
        assert!(res_mp.nodes > 50);
    }
}
