//! Pure-Rust UCI chess engine binary for Saccadion.
//! Supports full UCI protocol, live `.saccnet` weights loading, biological diagnostics (`eval`),
//! `perft`, and multi-position `bench`.

use saccadion_chess::features::extract_threat_masks;
use saccadion_chess::movegen::{generate_legal, parse_uci_move};
use saccadion_chess::perft::perft;
use saccadion_chess::position::Position;
use saccadion_chess::types::{Color, Move, MoveList, Square};
use saccadion_nn::config::{splitmix64, ModelConfig};
use saccadion_nn::stream_a::{evaluate_giant_fiber_cp, evaluate_giant_fiber_fp32, evaluate_stream_a};
use saccadion_nn::stream_b::{evaluate_stream_b, evaluate_stream_b_fp32};
use saccadion_nn::weights::SaccadionWeights;
use saccadion_search::{SearchConfig, SearchEngine, MATE_SCORE, MATE_THRESHOLD};
use std::fs::File;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn format_score_uci(score: i32) -> String {
    if score > MATE_THRESHOLD {
        let plies = (MATE_SCORE - score).max(1);
        format!("mate {}", (plies + 1) / 2)
    } else if score < -MATE_THRESHOLD {
        let plies = (MATE_SCORE + score).max(1);
        format!("mate -{}", (plies + 1) / 2)
    } else {
        format!("cp {}", score)
    }
}

fn print_bio_eval(engine: &mut SearchEngine, pos: &mut Position) {
    let mut legal = MoveList::new();
    generate_legal(pos, &mut legal);
    let out_a = evaluate_stream_a(
        pos,
        &engine.weights,
        &mut engine.ws_a,
        Some(&engine.plasticity),
        None,
        1.0,
    );
    let gf_cp = evaluate_giant_fiber_cp(pos, &engine.weights, &mut engine.ws_a);
    let novelty = engine
        .plasticity
        .mbon_alpha_prime3_novelty(&out_a.active_apbp_stm, false);
    let out_b = evaluate_stream_b(
        pos,
        &legal,
        &engine.weights,
        &out_a.mbon_summary,
        out_a.sez_cp,
        Some(&engine.plasticity),
        novelty,
        1.0,
        true,
    );
    let (sp, dp) = engine.weights.param_counts();
    println!("=== SACCADION-4 BIOLOGICAL CIRCUIT TELEMETRY ===");
    println!("FEN                : {}", pos.to_fen());
    println!("Params (Sparse/Dns): {} / {}", sp, dp);
    println!(
        "SEZ / Giant Fiber  : sez={:+.1} cp | giant_fiber={:+} cp",
        out_a.sez_cp, gf_cp
    );
    println!(
        "Stream A Eval (cp) : total={:+.1} (frozen={:+.1}, gamma={:+.1}, game={:+.1})",
        out_a.total_cp, out_a.slow_cp, out_a.gamma_cp, out_a.game_cp
    );
    println!(
        "Stream A APL NNZ   : {:.1}% active KCs",
        out_a.active_kc_ratio * 100.0
    );
    println!(
        "Lateral Horn Veto  : blunder={:.2} fortress={:.2} mate={:.2} passer={:.2}",
        out_a.veto.blunder_risk,
        out_a.veto.fortress_prob,
        out_a.veto.mate_danger,
        out_a.veto.passer_urgency
    );
    println!(
        "Stream B Eval (cp) : {:+.1} | CX Ticks={}/{} | PFL2 Sharpness={:.3} | Novelty={:.3}",
        out_b.deep_cp, out_b.ticks_used, engine.weights.cfg.cx_max_ticks, out_b.pfl2_sharpness, novelty
    );
    println!(
        "Pressure (10-20ply): mob_col={:+.2} ent={:+.2} king={:+.2} space={:+.2} proph={:+.2}",
        out_b.strategic_pressure[0],
        out_b.strategic_pressure[1],
        out_b.strategic_pressure[2],
        out_b.strategic_pressure[3],
        out_b.strategic_pressure[4]
    );
    println!("Top PFL3 Moves     :");
    for pm in out_b.policy_moves.iter().take(5) {
        println!(
            "  {}  prob={:.3}  noduli_expected_delta={:+.1} cp",
            pm.mv.to_uci(),
            pm.prob,
            pm.expected_delta_cp
        );
    }
}

fn run_bench(engine: &mut SearchEngine, depth: i32) {
    let fens = [
        saccadion_chess::position::STARTPOS_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "3r2k1/5ppp/8/8/8/8/5PPP/3RR1K1 w - - 0 1",
    ];
    let t0 = Instant::now();
    let mut total_nodes = 0u64;
    let mut total_sb = 0u64;
    for fen in fens {
        engine.new_game();
        let mut pos = Position::from_fen(fen).unwrap();
        let cfg = SearchConfig {
            max_depth: depth,
            ..SearchConfig::default()
        };
        let res = engine.search(&mut pos, cfg, |_| {});
        total_nodes += res.nodes;
        total_sb += res.stream_b_evals;
    }
    let elapsed = t0.elapsed().as_secs_f64().max(1e-6);
    println!(
        "Bench depth={} | nodes={} | nps={:.0} | stream_b_evals={} | time={:.3}s",
        depth,
        total_nodes,
        total_nodes as f64 / elapsed,
        total_sb,
        elapsed
    );
}

fn cp_to_wdl_probs(cp: f32) -> [f32; 3] {
    let w = 1.0 / (1.0 + (-(cp - 90.0) / 140.0).exp());
    let l = 1.0 / (1.0 + (-(-cp - 90.0) / 140.0).exp());
    let d = (1.0 - w - l).max(0.02);
    let s = w + d + l;
    [w / s, d / s, l / s]
}

fn pack_selfplay_record(
    pos: &Position,
    wdl: [f32; 3],
    best_mv: Move,
    next_mv: Move,
) -> [u8; 128] {
    let mut buf = [0u8; 128];
    for sq in 0..64 {
        let code: u8 = match pos.piece_at(Square::new(sq as u8)) {
            None => 0,
            Some((Color::White, p)) => 1 + p.index() as u8,
            Some((Color::Black, p)) => 7 + p.index() as u8,
        };
        if (sq & 1) == 0 {
            buf[sq >> 1] |= code & 0x0F;
        } else {
            buf[sq >> 1] |= (code & 0x0F) << 4;
        }
    }
    let stm_bit = (pos.side_to_move.index() as u32) & 1;
    let castling_bits = ((pos.castling.0 as u32) & 0x0F) << 1;
    let ep_bits = (pos.ep_square.map(|s| s.file() as u32 + 1).unwrap_or(0) & 0x0F) << 5;
    let halfmove_bits = ((pos.halfmove_clock as u32).min(255)) << 9;
    let phase_bits = ((pos.phase_bucket() as u32) & 0x07) << 17;
    let meta = stm_bit | castling_bits | ep_bits | halfmove_bits | phase_bits;
    buf[32..36].copy_from_slice(&meta.to_le_bytes());
    for i in 0..3 {
        buf[36 + i * 4..40 + i * 4].copy_from_slice(&wdl[i].to_le_bytes());
    }
    let b_u16 = ((best_mv.from_sq().index() as u16) << 6) | (best_mv.to_sq().index() as u16);
    let n_u16 = if !next_mv.is_null() {
        ((next_mv.from_sq().index() as u16) << 6) | (next_mv.to_sq().index() as u16)
    } else {
        b_u16
    };
    buf[48..50].copy_from_slice(&b_u16.to_le_bytes());
    buf[50..52].copy_from_slice(&n_u16.to_le_bytes());
    buf[52..54].copy_from_slice(&b_u16.to_le_bytes());
    buf[60..62].copy_from_slice(&65535u16.to_le_bytes());
    let (tm_s, tm_o) = extract_threat_masks(pos);
    buf[112..120].copy_from_slice(&tm_s.to_le_bytes());
    buf[120..128].copy_from_slice(&tm_o.to_le_bytes());
    buf
}

/// Pure-Rust zero-teacher dFB self-discovery generator: plays diverse multi-threaded self-play games
/// at `depth`, blends search eval + true game outcome, and writes a valid `.sacc` v1 binary shard!
fn run_selfplay(weights_path: &Path, out_path: &Path, num_games: usize, depth: i32) {
    let weights = Arc::new(SaccadionWeights::load_saccnet(weights_path).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {}", weights_path.display(), e);
        std::process::exit(1);
    }));
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 16))
        .unwrap_or(2)
        .min(num_games.max(1));
    let t0 = Instant::now();
    println!(
        "[dFB SelfPlay] Generating {} zero-teacher games at depth {} across {} threads...",
        num_games, depth, n_threads
    );

    let all_records: Vec<[u8; 128]> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(n_threads);
        for tid in 0..n_threads {
            let w_ref = Arc::clone(&weights);
            let games_for_t = (num_games + n_threads - 1 - tid) / n_threads;
            handles.push(scope.spawn(move || {
                let mut engine = SearchEngine::new((*w_ref).clone(), 16);
                let mut rng: u64 = 0x5ACC_8000_0000_0001 ^ ((tid as u64).wrapping_mul(0x9E37_79B9));
                let mut local_recs: Vec<[u8; 128]> = Vec::new();
                for _g in 0..games_for_t {
                    engine.new_game();
                    let mut pos = Position::startpos();
                    let open_plies = 4 + (splitmix64(&mut rng) as usize % 3);
                    for _ in 0..open_plies {
                        let mut legal = MoveList::new();
                        generate_legal(&mut pos, &mut legal);
                        if legal.is_empty() {
                            break;
                        }
                        let pick = (splitmix64(&mut rng) as usize) % legal.len();
                        pos.make_move(legal.as_slice()[pick]);
                    }
                    let mut samples: Vec<(Position, Move, i32, Color)> = Vec::new();
                    let mut white_outcome = 0.5f32;
                    for _ply in 0..160 {
                        if pos.halfmove_clock >= 100 || pos.is_insufficient_material() || pos.is_repetition() {
                            white_outcome = 0.5;
                            break;
                        }
                        let mut legal = MoveList::new();
                        generate_legal(&mut pos, &mut legal);
                        if legal.is_empty() {
                            white_outcome = if pos.in_check() {
                                if pos.side_to_move == Color::White { 0.0 } else { 1.0 }
                            } else {
                                0.5
                            };
                            break;
                        }
                        let scfg = SearchConfig {
                            max_depth: depth,
                            threads: 1,
                            use_stream_b: true,
                            ..SearchConfig::default()
                        };
                        let res = engine.search(&mut pos, scfg, |_| {});
                        let mv = res.best_move;
                        if mv.is_null() {
                            break;
                        }
                        let is_quiet = !pos.in_check() && !mv.is_capture() && !mv.is_promotion();
                        if is_quiet && res.score_cp.abs() < 1500 {
                            samples.push((pos.clone(), mv, res.score_cp, pos.side_to_move));
                        }
                        if res.score_cp.abs() > MATE_THRESHOLD - 100 {
                            white_outcome = if (res.score_cp > 0) == (pos.side_to_move == Color::White) {
                                1.0
                            } else {
                                0.0
                            };
                            break;
                        }
                        pos.make_move(mv);
                    }
                    for i in 0..samples.len() {
                        let (ref p, mv, sc, stm) = samples[i];
                        let s_wdl = cp_to_wdl_probs(sc as f32);
                        let g_w = if stm == Color::White { white_outcome } else { 1.0 - white_outcome };
                        let g_wdl = if g_w > 0.75 {
                            [1.0, 0.0, 0.0]
                        } else if g_w < 0.25 {
                            [0.0, 0.0, 1.0]
                        } else {
                            [0.0, 1.0, 0.0]
                        };
                        let blend = [
                            0.65 * s_wdl[0] + 0.35 * g_wdl[0],
                            0.65 * s_wdl[1] + 0.35 * g_wdl[1],
                            0.65 * s_wdl[2] + 0.35 * g_wdl[2],
                        ];
                        let next_mv = samples
                            .get(i + 2)
                            .filter(|s2| s2.3 == stm)
                            .map(|s2| s2.1)
                            .unwrap_or(Move::NULL);
                        local_recs.push(pack_selfplay_record(p, blend, mv, next_mv));
                    }
                }
                local_recs
            }));
        }
        let mut merged = Vec::new();
        for h in handles {
            merged.extend(h.join().unwrap_or_default());
        }
        merged
    });

    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let f = File::create(out_path).unwrap_or_else(|e| {
        eprintln!("Failed to create {}: {}", out_path.display(), e);
        std::process::exit(1);
    });
    let mut bw = BufWriter::new(f);
    bw.write_all(b"SACC0001").unwrap();
    bw.write_all(&0u32.to_le_bytes()).unwrap();
    bw.write_all(&(all_records.len() as u32).to_le_bytes()).unwrap();
    for rec in &all_records {
        bw.write_all(rec).unwrap();
    }
    bw.flush().unwrap();
    let secs = t0.elapsed().as_secs_f64().max(1e-6);
    println!(
        "[dFB SelfPlay] Wrote {} quiet positions from {} games in {:.2}s ({:.0} pos/s) -> {}",
        all_records.len(),
        num_games,
        secs,
        all_records.len() as f64 / secs,
        out_path.display()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Pure-Rust zero-teacher dFB self-play shard generator:
    // `saccadion selfplay <weights.saccnet> <out.sacc> [num_games] [depth]`
    if args.len() >= 4 && args[1] == "selfplay" {
        let weights_path = Path::new(&args[2]);
        let out_path = Path::new(&args[3]);
        let num_games: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
        let depth: i32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(6);
        run_selfplay(weights_path, out_path, num_games, depth);
        return;
    }

    // Optional direct CLI mode: `saccadion parity-check <weights.saccnet> [fen]` for JAX<->Rust parity!
    if args.len() >= 3 && args[1] == "parity-check" {
        let weights_path = Path::new(&args[2]);
        let weights = SaccadionWeights::load_saccnet(weights_path).unwrap_or_else(|e| {
            eprintln!("Failed to load {}: {}", weights_path.display(), e);
            std::process::exit(1);
        });
        let fen = args
            .get(3)
            .map(|s| s.as_str())
            .unwrap_or(saccadion_chess::position::STARTPOS_FEN);
        let mut pos = Position::from_fen(fen).unwrap();
        let mut legal = MoveList::new();
        generate_legal(&mut pos, &mut legal);
        let mut ws = saccadion_nn::StreamAWorkspace::new(&weights);
        let gf_fp32 = evaluate_giant_fiber_fp32(&pos, &weights, &mut ws);
        let gf_i16 = evaluate_giant_fiber_cp(&pos, &weights, &mut ws);
        let out_a = evaluate_stream_a(&pos, &weights, &mut ws, None, None, 1.0);
        let out_b = evaluate_stream_b_fp32(
            &pos,
            &legal,
            &weights,
            &out_a.mbon_summary,
            out_a.sez_cp,
            None,
            0.0,
            1.0,
            false,
        );
        let out_b_i8 = evaluate_stream_b(
            &pos,
            &legal,
            &weights,
            &out_a.mbon_summary,
            out_a.sez_cp,
            None,
            0.0,
            1.0,
            false,
        );
        let retro_mean = out_b.retro_goal_saliency.iter().sum::<f32>() / 64.0;
        println!(
            "PARITY gf_fp32={:.6} gf_i16={} stream_a_slow_cp={:.6} stream_a_wdl={:.6},{:.6},{:.6} stream_b_deep_cp={:.6} stream_b_i8_cp={:.6} retro_goal_mean={:.6} stream_b_wdl={:.6},{:.6},{:.6} pfl2={:.6} press={:.6},{:.6},{:.6},{:.6},{:.6}",
            gf_fp32,
            gf_i16,
            out_a.slow_cp,
            out_a.wdl_probs[0],
            out_a.wdl_probs[1],
            out_a.wdl_probs[2],
            out_b.deep_cp,
            out_b_i8.deep_cp,
            retro_mean,
            out_b.wdl_probs[0],
            out_b.wdl_probs[1],
            out_b.wdl_probs[2],
            out_b.pfl2_sharpness,
            out_b.strategic_pressure[0],
            out_b.strategic_pressure[1],
            out_b.strategic_pressure[2],
            out_b.strategic_pressure[3],
            out_b.strategic_pressure[4],
        );
        return;
    }

    // Auto-load `saccadion_best.saccnet` from CWD, executable folder, or repo root (../../)
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    let mut weight_cands = vec![Path::new("saccadion_best.saccnet").to_path_buf()];
    if let Some(ref d) = exe_dir {
        weight_cands.push(d.join("saccadion_best.saccnet"));
        if let Some(p1) = d.parent() {
            weight_cands.push(p1.join("saccadion_best.saccnet"));
            if let Some(p2) = p1.parent() {
                weight_cands.push(p2.join("saccadion_best.saccnet"));
            }
        }
    }
    let auto_weights = weight_cands
        .into_iter()
        .find(|p| p.exists())
        .and_then(|p| SaccadionWeights::load_saccnet(&p).ok());

    let weights = auto_weights.unwrap_or_else(|| SaccadionWeights::init_random(ModelConfig::tiny(), 0x5ACC_AD10));
    let mut hash_mb = 64usize;
    let default_threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 4))
        .unwrap_or(1);
    let mut engine = SearchEngine::new(weights, hash_mb);
    let mut pos = Position::startpos();
    let mut base_cfg = SearchConfig {
        threads: default_threads,
        ..SearchConfig::default()
    };

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line_res in stdin.lock().lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        match tokens[0] {
            "uci" => {
                println!("id name Saccadion 8.0 (Drosophila Retrograde-Fischer-Graph+INT8)");
                println!("id author Saccadion Research Team");
                println!("option name Hash type spin default {} min 1 max 4096", hash_mb);
                println!("option name Threads type spin default {} min 1 max 64", default_threads);
                println!("option name WeightsPath type string default ");
                println!("option name AdaptivePlasticity type check default true");
                println!("option name UseStreamB type check default true");
                println!("option name Serotonin type spin default 100 min 50 max 150");
                println!("option name Octopamine type spin default 100 min 70 max 140");
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                engine.new_game();
                pos = Position::startpos();
            }
            "setoption" => {
                // Parse `setoption name <N> value <V>`
                if let Some(n_pos) = tokens.iter().position(|&t| t == "name") {
                    let v_pos = tokens.iter().position(|&t| t == "value");
                    let name = match v_pos {
                        Some(vp) => tokens[n_pos + 1..vp].join(" "),
                        None => tokens[n_pos + 1..].join(" "),
                    };
                    let val = v_pos.map(|vp| tokens[vp + 1..].join(" ")).unwrap_or_default();
                    match name.as_str() {
                        "Hash" => {
                            if let Ok(mb) = val.parse::<usize>() {
                                hash_mb = mb.clamp(1, 4096);
                                engine.resize_tt(hash_mb);
                            }
                        }
                        "Threads" => {
                            if let Ok(t) = val.parse::<usize>() {
                                base_cfg.threads = t.clamp(1, 64);
                            }
                        }
                        "WeightsPath" if !val.is_empty() => {
                            match SaccadionWeights::load_saccnet(Path::new(&val)) {
                                Ok(w) => {
                                    engine = SearchEngine::new(w, hash_mb);
                                    println!("info string Loaded .saccnet weights from {}", val);
                                }
                                Err(e) => {
                                    println!("info string Error loading weights {}: {}", val, e);
                                }
                            }
                        }
                        "AdaptivePlasticity" => {
                            base_cfg.use_dopamine = val == "true";
                        }
                        "UseStreamB" => {
                            base_cfg.use_stream_b = val == "true";
                        }
                        "Serotonin" => {
                            if let Ok(v) = val.parse::<f32>() {
                                base_cfg.serotonin = (v / 100.0).clamp(0.5, 1.5);
                            }
                        }
                        "Octopamine" => {
                            if let Ok(v) = val.parse::<f32>() {
                                base_cfg.octopamine = (v / 100.0).clamp(0.7, 1.4);
                            }
                        }
                        _ => {}
                    }
                }
            }
            "position" => {
                let mut idx = 1usize;
                if idx < tokens.len() && tokens[idx] == "startpos" {
                    pos = Position::startpos();
                    idx += 1;
                } else if idx < tokens.len() && tokens[idx] == "fen" {
                    idx += 1;
                    let mut fen_parts = Vec::new();
                    while idx < tokens.len() && tokens[idx] != "moves" {
                        fen_parts.push(tokens[idx]);
                        idx += 1;
                    }
                    if let Ok(p) = Position::from_fen(&fen_parts.join(" ")) {
                        pos = p;
                    }
                }
                if idx < tokens.len() && tokens[idx] == "moves" {
                    for &m_str in &tokens[idx + 1..] {
                        if let Some(mv) = parse_uci_move(&mut pos, m_str) {
                            pos.make_move(mv);
                        }
                    }
                }
            }
            "go" => {
                let mut cfg = base_cfg.clone();
                cfg.max_depth = 64;
                cfg.max_nodes = u64::MAX;
                cfg.time_limit_ms = 0;

                let mut wtime = 0u64;
                let mut btime = 0u64;
                let mut winc = 0u64;
                let mut binc = 0u64;
                let mut i = 1usize;
                while i < tokens.len() {
                    match tokens[i] {
                        "depth" if i + 1 < tokens.len() => {
                            cfg.max_depth = tokens[i + 1].parse().unwrap_or(6);
                            i += 2;
                        }
                        "movetime" if i + 1 < tokens.len() => {
                            cfg.time_limit_ms = tokens[i + 1].parse().unwrap_or(1000);
                            i += 2;
                        }
                        "nodes" if i + 1 < tokens.len() => {
                            cfg.max_nodes = tokens[i + 1].parse().unwrap_or(u64::MAX);
                            i += 2;
                        }
                        "wtime" if i + 1 < tokens.len() => {
                            wtime = tokens[i + 1].parse().unwrap_or(0);
                            i += 2;
                        }
                        "btime" if i + 1 < tokens.len() => {
                            btime = tokens[i + 1].parse().unwrap_or(0);
                            i += 2;
                        }
                        "winc" if i + 1 < tokens.len() => {
                            winc = tokens[i + 1].parse().unwrap_or(0);
                            i += 2;
                        }
                        "binc" if i + 1 < tokens.len() => {
                            binc = tokens[i + 1].parse().unwrap_or(0);
                            i += 2;
                        }
                        _ => i += 1,
                    }
                }
                if cfg.time_limit_ms == 0 && (wtime > 0 || btime > 0) {
                    let (rem, inc) = if pos.side_to_move == Color::White {
                        (wtime, winc)
                    } else {
                        (btime, binc)
                    };
                    cfg.time_limit_ms = (rem / 28 + (inc * 3) / 4).max(20);
                }

                let res = engine.search(&mut pos, cfg, |info| {
                    let nps = if info.elapsed_ms > 0 {
                        (info.nodes * 1000) / info.elapsed_ms
                    } else {
                        info.nodes
                    };
                    let pv_str: Vec<String> = info.pv.iter().map(|m| m.to_uci()).collect();
                    println!(
                        "info depth {} score {} nodes {} nps {} time {} pv {}",
                        info.depth,
                        format_score_uci(info.score_cp),
                        info.nodes,
                        nps,
                        info.elapsed_ms,
                        pv_str.join(" ")
                    );
                });
                println!("bestmove {}", res.best_move.to_uci());
            }
            "eval" => print_bio_eval(&mut engine, &mut pos),
            "perft" => {
                let d: u32 = tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
                let t0 = Instant::now();
                let n = perft(&mut pos, d);
                let secs = t0.elapsed().as_secs_f64().max(1e-6);
                println!("perft({}) = {} ({:.0} nps)", d, n, n as f64 / secs);
            }
            "bench" => {
                let d: i32 = tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
                run_bench(&mut engine, d);
            }
            "stop" => {}
            "quit" => break,
            _ => {}
        }
        let _ = stdout.flush();
    }
}
