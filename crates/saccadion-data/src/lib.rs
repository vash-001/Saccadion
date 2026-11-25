//! Saccadion Strategic Data Forge crate & CLI tool.

pub mod format;
pub mod pgn;
pub mod pipeline;

#[cfg(test)]
mod tests {
    use crate::format::{read_sacc_shard, write_sacc_shard, SACC_RECORD_BYTES};
    use crate::pgn::parse_pgn_text;
    use crate::pipeline::{extract_trajectory_windows, game_to_sacc_records};

    const SAMPLE_PGN: &str = r#"
[Event "Strategic Squeeze Demo"]
[White "Saccadion"]
[Black "Opponent"]
[Result "1-0"]

1. d4 { [%eval 0.25] } 1... Nf6 { [%eval 0.22] } 2. c4 { [%eval 0.28] } 2... e6 { [%eval 0.30] }
3. Nc3 { [%eval 0.32] } 3... Bb4 { [%eval 0.25] } 4. Qc2 { [%eval 0.35] } 4... O-O { [%eval 0.30] }
5. a3 { [%eval 0.40] } 5... Bxc3+ { [%eval 0.38] } 6. Qxc3 { [%eval 0.42] } 6... b6 { [%eval 0.45] }
7. Bg5 { [%eval 0.50] } 7... Bb7 { [%eval 0.48] } 8. e3 { [%eval 0.52] } 8... d6 { [%eval 0.55] }
9. f3 { [%eval 0.65] } 9... Nbd7 { [%eval 0.60] } 10. Bd3 { [%eval 0.72] } 10... c5 { [%eval 0.75] }
1-0
"#;

    #[test]
    fn test_pgn_to_strategic_pressure_sacc_roundtrip() {
        let games = parse_pgn_text(SAMPLE_PGN);
        assert_eq!(games.len(), 1);
        assert_eq!(games[0].moves.len(), 20);
        assert_eq!(games[0].white_result, 1.0);

        let records = game_to_sacc_records(&games[0]);
        assert_eq!(records.len(), 20);
        assert_eq!(records[0].to_bytes().len(), SACC_RECORD_BYTES);

        // Verify trajectory windowing for Stage-3 Meta-Dopamine (K=12)
        let traj = extract_trajectory_windows(&records, 12);
        assert!(traj.len() >= 12 && traj.len() % 12 == 0);

        let tmp = std::env::temp_dir().join("saccadion_data_roundtrip.sacc");
        write_sacc_shard(&tmp, &records, false).expect("write shard");
        let (is_traj, loaded) = read_sacc_shard(&tmp).expect("read shard");
        let _ = std::fs::remove_file(&tmp);
        assert!(!is_traj);
        assert_eq!(loaded, records);
    }
}
