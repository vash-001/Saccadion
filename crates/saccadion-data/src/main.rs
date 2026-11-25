//! CLI binary for packing PGN databases into organized train/val/traj 128-byte `.sacc` shards
//! and ultra-fast multi-threaded pre-decoding into memmap tensor caches for Colab TPU training.

use saccadion_chess::features::{
    extract_bilateral_sparse, extract_lateral_horn_features, extract_stream_b_board,
    LATERAL_HORN_DIM, STREAM_B_SQ_FEAT_DIM,
};
use saccadion_data::format::{f32_to_f16, read_sacc_shard, write_sacc_shard, SaccRecord};
use saccadion_data::pgn::parse_pgn_text;
use saccadion_data::pipeline::{extract_trajectory_windows, game_to_sacc_records};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_SPARSE_CACHE: usize = 80;

fn read_optional_concepts(shard_path: &Path, expected_len: usize) -> Vec<u16> {
    let mut name = shard_path.file_name().unwrap_or_default().to_os_string();
    name.push(".concepts.u16");
    let cpath = shard_path.with_file_name(name);
    if let Ok(mut f) = File::open(&cpath) {
        let mut buf = vec![0u8; expected_len * 2];
        if f.read_exact(&mut buf).is_ok() {
            let mut out = Vec::with_capacity(expected_len);
            for chunk in buf.chunks_exact(2) {
                out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            return out;
        }
    }
    vec![0u16; expected_len]
}

fn write_raw_slice<T>(path: &Path, data: &[T]) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, f);
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    w.write_all(bytes)?;
    w.flush()
}

fn run_predecode(out_cache_dir: &Path, shard_paths: &[PathBuf]) {
    let t0 = Instant::now();
    fs::create_dir_all(out_cache_dir).expect("create cache dir");

    let mut all_recs: Vec<SaccRecord> = Vec::new();
    let mut all_concepts: Vec<u16> = Vec::new();
    for p in shard_paths {
        let (_is_traj, recs) = read_sacc_shard(p).unwrap_or_else(|e| {
            eprintln!("Failed to read shard {}: {}", p.display(), e);
            std::process::exit(1);
        });
        let concepts = read_optional_concepts(p, recs.len());
        all_recs.extend(recs);
        all_concepts.extend(concepts);
    }

    let n = all_recs.len();
    if n == 0 {
        eprintln!("No records found to predecode.");
        std::process::exit(1);
    }

    let mut sp_stm = vec![-1i16; n * MAX_SPARSE_CACHE];
    let mut sp_opp = vec![-1i16; n * MAX_SPARSE_CACHE];
    let mut board_f16 = vec![0u16; n * 64 * STREAM_B_SQ_FEAT_DIM];
    let mut lh_feat = vec![0.0f32; n * LATERAL_HORN_DIM];
    let mut phase = vec![0u8; n];
    let mut wdl = vec![0.0f32; n * 3];
    let mut top4_m = vec![0u16; n * 4];
    let mut top4_p = vec![0.0f32; n * 4];
    let mut next_mv = vec![0u16; n];
    let mut pressure = vec![0.0f32; n * 5];
    let mut lh_targets = vec![0.0f32; n * 4];
    let mut vol = vec![0.0f32; n];
    let mut threat_64 = vec![0u8; n * 64];

    let n_threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(4)
        .min(n.max(1));
    let chunk = (n + n_threads - 1) / n_threads;

    std::thread::scope(|scope| {
        let mut rem_sp_s = sp_stm.as_mut_slice();
        let mut rem_sp_o = sp_opp.as_mut_slice();
        let mut rem_bd = board_f16.as_mut_slice();
        let mut rem_lh = lh_feat.as_mut_slice();
        let mut rem_ph = phase.as_mut_slice();
        let mut rem_wdl = wdl.as_mut_slice();
        let mut rem_t4m = top4_m.as_mut_slice();
        let mut rem_t4p = top4_p.as_mut_slice();
        let mut rem_nm = next_mv.as_mut_slice();
        let mut rem_pr = pressure.as_mut_slice();
        let mut rem_lht = lh_targets.as_mut_slice();
        let mut rem_vol = vol.as_mut_slice();
        let mut rem_th = threat_64.as_mut_slice();

        for start in (0..n).step_by(chunk) {
            let len = (n - start).min(chunk);
            let recs_slice = &all_recs[start..start + len];

            let (c_sp_s, rest) = rem_sp_s.split_at_mut(len * MAX_SPARSE_CACHE);
            rem_sp_s = rest;
            let (c_sp_o, rest) = rem_sp_o.split_at_mut(len * MAX_SPARSE_CACHE);
            rem_sp_o = rest;
            let (c_bd, rest) = rem_bd.split_at_mut(len * 64 * STREAM_B_SQ_FEAT_DIM);
            rem_bd = rest;
            let (c_lh, rest) = rem_lh.split_at_mut(len * LATERAL_HORN_DIM);
            rem_lh = rest;
            let (c_ph, rest) = rem_ph.split_at_mut(len);
            rem_ph = rest;
            let (c_wdl, rest) = rem_wdl.split_at_mut(len * 3);
            rem_wdl = rest;
            let (c_t4m, rest) = rem_t4m.split_at_mut(len * 4);
            rem_t4m = rest;
            let (c_t4p, rest) = rem_t4p.split_at_mut(len * 4);
            rem_t4p = rest;
            let (c_nm, rest) = rem_nm.split_at_mut(len);
            rem_nm = rest;
            let (c_pr, rest) = rem_pr.split_at_mut(len * 5);
            rem_pr = rest;
            let (c_lht, rest) = rem_lht.split_at_mut(len * 4);
            rem_lht = rest;
            let (c_vol, rest) = rem_vol.split_at_mut(len);
            rem_vol = rest;
            let (c_th, rest) = rem_th.split_at_mut(len * 64);
            rem_th = rest;

            scope.spawn(move || {
                for (i, rec) in recs_slice.iter().enumerate() {
                    let pos = rec.to_position();
                    let (fs, fo) = extract_bilateral_sparse(&pos);
                    let dst_s = &mut c_sp_s[i * MAX_SPARSE_CACHE..(i + 1) * MAX_SPARSE_CACHE];
                    for (k, &idx) in fs.as_slice().iter().take(MAX_SPARSE_CACHE).enumerate() {
                        dst_s[k] = idx as i16;
                    }
                    let dst_o = &mut c_sp_o[i * MAX_SPARSE_CACHE..(i + 1) * MAX_SPARSE_CACHE];
                    for (k, &idx) in fo.as_slice().iter().take(MAX_SPARSE_CACHE).enumerate() {
                        dst_o[k] = idx as i16;
                    }

                    let bd = extract_stream_b_board(&pos);
                    let dst_bd =
                        &mut c_bd[i * 64 * STREAM_B_SQ_FEAT_DIM..(i + 1) * 64 * STREAM_B_SQ_FEAT_DIM];
                    for sq in 0..64 {
                        let base = sq * STREAM_B_SQ_FEAT_DIM;
                        for f in 0..STREAM_B_SQ_FEAT_DIM {
                            dst_bd[base + f] = f32_to_f16(bd[sq][f]);
                        }
                    }

                    let lh = extract_lateral_horn_features(&pos);
                    c_lh[i * LATERAL_HORN_DIM..(i + 1) * LATERAL_HORN_DIM].copy_from_slice(&lh);

                    c_ph[i] = ((rec.meta_flags >> 17) & 0x07) as u8;
                    c_wdl[i * 3..(i + 1) * 3].copy_from_slice(&rec.wdl_target);
                    c_t4m[i * 4..(i + 1) * 4].copy_from_slice(&rec.top4_moves);
                    let mut psum = 0.0f32;
                    let mut pf = [0.0f32; 4];
                    for k in 0..4 {
                        pf[k] = (rec.top4_probs_u16[k] as f32) / 65535.0;
                        psum += pf[k];
                    }
                    let inv = 1.0 / psum.max(1e-6);
                    for k in 0..4 {
                        c_t4p[i * 4 + k] = pf[k] * inv;
                    }
                    c_nm[i] = rec.next_move_u16;
                    c_pr[i * 5..(i + 1) * 5].copy_from_slice(&rec.pressure_targets);
                    c_lht[i * 4..(i + 1) * 4].copy_from_slice(&rec.lh_targets);
                    c_vol[i] = rec.volatility;
                    let tm = rec.threat_mask_stm;
                    for sq in 0..64 {
                        c_th[i * 64 + sq] = if (tm & (1u64 << sq)) != 0 { 1 } else { 0 };
                    }
                }
            });
        }
    });

    write_raw_slice(&out_cache_dir.join("sparse_stm.i16"), &sp_stm).expect("write sparse_stm");
    write_raw_slice(&out_cache_dir.join("sparse_opp.i16"), &sp_opp).expect("write sparse_opp");
    write_raw_slice(&out_cache_dir.join("board_feats.f16"), &board_f16).expect("write board_feats");
    write_raw_slice(&out_cache_dir.join("lh_feat.f32"), &lh_feat).expect("write lh_feat");
    write_raw_slice(&out_cache_dir.join("phase.u8"), &phase).expect("write phase");
    write_raw_slice(&out_cache_dir.join("wdl.f32"), &wdl).expect("write wdl");
    write_raw_slice(&out_cache_dir.join("top4_moves.u16"), &top4_m).expect("write top4_moves");
    write_raw_slice(&out_cache_dir.join("top4_probs.f32"), &top4_p).expect("write top4_probs");
    write_raw_slice(&out_cache_dir.join("next_move.u16"), &next_mv).expect("write next_move");
    write_raw_slice(&out_cache_dir.join("pressure.f32"), &pressure).expect("write pressure");
    write_raw_slice(&out_cache_dir.join("lh_targets.f32"), &lh_targets).expect("write lh_targets");
    write_raw_slice(&out_cache_dir.join("vol.f32"), &vol).expect("write vol");
    write_raw_slice(&out_cache_dir.join("threat_64.u8"), &threat_64).expect("write threat_64");
    write_raw_slice(&out_cache_dir.join("causal_bits.u16"), &all_concepts).expect("write causal_bits");

    let meta_json = format!(
        "{{\"count\": {}, \"max_sparse\": {}, \"version\": 2, \"num_shards\": {}}}",
        n,
        MAX_SPARSE_CACHE,
        shard_paths.len()
    );
    fs::write(out_cache_dir.join("meta.json"), meta_json).expect("write meta.json");

    let dt = t0.elapsed().as_secs_f64();
    println!(
        "Rust predecoded {} records across {} threads in {:.2}s ({:.0} pos/s) -> {}",
        n,
        n_threads,
        dt,
        n as f64 / dt.max(1e-6),
        out_cache_dir.display()
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage:");
        eprintln!("  saccadion-data pack <input.pgn> <output.sacc> [--traj]");
        eprintln!("  saccadion-data organize <input.pgn> <out_dir> [--val-pct 5] [--max-games 20000]");
        eprintln!("  saccadion-data predecode <out_cache_dir> <shard1.sacc> [shard2.sacc ...]");
        std::process::exit(1);
    }

    let mode = args[1].as_str();
    if mode == "predecode" {
        let out_cache = Path::new(&args[2]);
        let shards: Vec<PathBuf> = args[3..].iter().map(PathBuf::from).collect();
        run_predecode(out_cache, &shards);
        return;
    }

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
