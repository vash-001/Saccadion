//! Pure-Rust UCI chess engine binary for Saccadion.
//! Supports full UCI protocol, live `.saccnet` weights loading, biological diagnostics (`eval`),
//! `perft`, and multi-position `bench`.

use saccadion_chess::movegen::{generate_legal, parse_uci_move};
use saccadion_chess::perft::perft;
use saccadion_chess::position::Position;
use saccadion_chess::types::{Color, MoveList};
use saccadion_nn::config::ModelConfig;
use saccadion_nn::stream_a::{evaluate_giant_fiber_cp, evaluate_stream_a};
use saccadion_nn::stream_b::evaluate_stream_b;
use saccadion_nn::weights::SaccadionWeights;
use saccadion_search::{SearchConfig, SearchEngine, MATE_SCORE, MATE_THRESHOLD};
use std::io::{self, BufRead, Write};
use std::path::Path;
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

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Optional direct CLI mode: `saccadion eval-saccnet <weights.saccnet> [fen]` for JAX<->Rust parity!
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
        let out_a = evaluate_stream_a(&pos, &weights, &mut ws, None, None, 1.0);
        let out_b = evaluate_stream_b(
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
        println!(
            "PARITY stream_a_slow_cp={:.6} stream_a_wdl={:.6},{:.6},{:.6} stream_b_deep_cp={:.6} stream_b_wdl={:.6},{:.6},{:.6} pfl2={:.6} press={:.6},{:.6},{:.6},{:.6},{:.6}",
            out_a.slow_cp,
            out_a.wdl_probs[0],
            out_a.wdl_probs[1],
            out_a.wdl_probs[2],
            out_b.deep_cp,
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

    // Auto-load `saccadion_best.saccnet` if present in CWD or alongside the binary, else Tiny fallback
    let auto_weights = [
        Path::new("saccadion_best.saccnet").to_path_buf(),
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("saccadion_best.saccnet")))
            .unwrap_or_default(),
    ]
    .into_iter()
    .find(|p| !p.as_os_str().is_empty() && p.exists())
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
                println!("id name Saccadion 5.0 (Drosophila 3-Lobe+Amacrine+INT16)");
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
