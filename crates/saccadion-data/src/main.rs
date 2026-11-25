//! CLI binary for packing PGN databases into organized train/val/traj 128-byte `.sacc` shards.

use saccadion_data::format::write_sacc_shard;
use saccadion_data::pgn::parse_pgn_text;
use saccadion_data::pipeline::{extract_trajectory_windows, game_to_sacc_records};
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage:");
        eprintln!("  saccadion-data pack <input.pgn> <output.sacc> [--traj]");
        eprintln!("  saccadion-data organize <input.pgn> <out_dir> [--val-pct 5] [--max-games 20000]");
        std::process::exit(1);
    }

    let mode = args[1].as_str();
    let input = Path::new(&args[2]);
    let pgn_text = fs::read_to_string(input).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", input.display(), e);
        std::process::exit(1);
    });
    let mut games = parse_pgn_text(&pgn_text);

    if mode == "pack" {
        let output = Path::new(&args[3]);
        let is_traj = args.iter().any(|a| a == "--traj");
        let mut all_records = Vec::new();
        for g in &games {
            let recs = game_to_sacc_records(g);
            if is_traj {
                all_records.extend(extract_trajectory_windows(&recs, 12));
            } else {
                all_records.extend(recs);
            }
        }
        write_sacc_shard(output, &all_records, is_traj).unwrap_or_else(|e| {
            eprintln!("Failed to write {}: {}", output.display(), e);
            std::process::exit(1);
        });
        println!(
            "Packed {} games -> {} records ({} KB) into {}",
            games.len(),
            all_records.len(),
            (all_records.len() * 128) / 1024,
            output.display()
        );
    } else if mode == "organize" {
        let out_dir = Path::new(&args[3]);
        fs::create_dir_all(out_dir).expect("create output shard dir");
        let mut val_pct: usize = 5;
        let mut max_games: usize = usize::MAX;
        let mut i = 4usize;
        while i < args.len() {
            match args[i].as_str() {
                "--val-pct" if i + 1 < args.len() => {
                    val_pct = args[i + 1].parse().unwrap_or(5).clamp(1, 30);
                    i += 2;
                }
                "--max-games" if i + 1 < args.len() => {
                    max_games = args[i + 1].parse().unwrap_or(usize::MAX);
                    i += 2;
                }
                _ => i += 1,
            }
        }
        if games.len() > max_games {
            games.truncate(max_games);
        }
        let val_count = ((games.len() * val_pct) / 100).max(1).min(games.len().saturating_sub(1));
        let split_idx = games.len().saturating_sub(val_count);
        let (train_games, val_games) = games.split_at(split_idx);

        let mut train_rand = Vec::new();
        let mut train_traj = Vec::new();
        for g in train_games {
            let recs = game_to_sacc_records(g);
            train_traj.extend(extract_trajectory_windows(&recs, 12));
            train_rand.extend(recs);
        }
        let mut val_rand = Vec::new();
        for g in val_games {
            val_rand.extend(game_to_sacc_records(g));
        }

        // Write cleanly chunked 100k-record shards + held-out validation shard
        let chunk_size = 100_000usize;
        for (idx, chunk) in train_rand.chunks(chunk_size).enumerate() {
            let p = out_dir.join(format!("train_random_{:03}.sacc", idx));
            write_sacc_shard(&p, chunk, false).expect("write train_random shard");
        }
        for (idx, chunk) in train_traj.chunks(chunk_size * 12 / 10).enumerate() {
            // Keep exact multiples of 12 for K=12 trajectory windows
            let usable = (chunk.len() / 12) * 12;
            if usable > 0 {
                let p = out_dir.join(format!("train_traj_{:03}.sacc", idx));
                write_sacc_shard(&p, &chunk[..usable], true).expect("write train_traj shard");
            }
        }
        if !val_rand.is_empty() {
            let p = out_dir.join("val_random.sacc");
            write_sacc_shard(&p, &val_rand, false).expect("write val_random shard");
        }

        println!(
            "Organized {} games (train={}, val={}) -> train_pos={}, val_pos={}, traj_windows={} in {}",
            games.len(),
            train_games.len(),
            val_games.len(),
            train_rand.len(),
            val_rand.len(),
            train_traj.len() / 12,
            out_dir.display()
        );
    } else {
        eprintln!("Unknown mode: {}", mode);
        std::process::exit(1);
    }
}
