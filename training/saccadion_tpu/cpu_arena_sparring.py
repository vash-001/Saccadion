"""Background CPU Stockfish Sparring Arena & Real-Elo Calibrator for Colab TPU VMs.

Exploits idle CPU cores in the background while the TPU trains:
1. Plays rapid UCI games between our latest Pure-Rust `saccadion` engine (auto-reloading
   `saccadion_latest.saccnet` after each epoch!) and calibrated Stockfish strength levels
   (plus an internal ~900 Elo material baseline for Epoch-1 calibration).
2. Auto-adjusts opponent difficulty upward as Saccadion improves (900 -> 1320 -> 1550 ->
   1850 -> 2200 -> 2600 -> 3000+ Elo) and computes Maximum-Likelihood Real Elo + 95% CI.
3. Harvests turning-point / loss positions against Stockfish (with Stockfish's refutation move
   and evaluation) into `sparring_hard.sacc` so future epochs actively learn anti-Stockfish play!
"""

from __future__ import annotations

import json
import math
import random
import shutil
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dataset import SACC_RECORD_DTYPE, SACC_SHARD_MAGIC

OPPONENT_LADDER: List[Tuple[str, int, Optional[int]]] = [
    # (display_name, calibrated_elo, stockfish_uci_elo_or_None_for_builtin_baseline)
    ("Baseline-Material-d2", 900, None),
    ("Stockfish-1320 (Lv0)", 1320, 1320),
    ("Stockfish-1550 (Lv3)", 1550, 1550),
    ("Stockfish-1850 (Lv6)", 1850, 1850),
    ("Stockfish-2200 (Lv10)", 2200, 2200),
    ("Stockfish-2600 (Lv15)", 2600, 2600),
    ("Stockfish-3000 (Lv20)", 3000, 3000),
]

PIECE_VALUES_CP = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}


def estimate_mle_elo(results: List[Tuple[int, float]]) -> Tuple[float, float]:
    """Computes logistic MLE Real Elo and approximate standard error from (opp_elo, score) pairs."""
    if not results:
        return 0.0, 999.0
    # Regularized Newton-Raphson with mild Gaussian prior around 1200 Elo (sigma=600)
    # so even 100% wins or 100% losses produce stable finite Elo estimates!
    r = 1200.0
    prior_mu = 1200.0
    prior_inv_var = 1.0 / (600.0 * 600.0)
    k = math.log(10.0) / 400.0
    for _ in range(25):
        grad = -(r - prior_mu) * prior_inv_var
        hess = -prior_inv_var
        for opp_elo, score in results:
            p = 1.0 / (1.0 + math.pow(10.0, (opp_elo - r) / 400.0))
            grad += k * (score - p)
            hess -= k * k * p * (1.0 - p)
        step = max(-180.0, min(180.0, -grad / min(hess, -1e-9)))
        r = max(400.0, min(3800.0, r + step))
        if abs(step) < 0.1:
            break
    se = 1.0 / math.sqrt(max(-hess, 1e-9))
    return float(r), float(min(450.0, 1.96 * se))


def find_or_install_stockfish() -> Optional[Path]:
    """Locates Stockfish binary, installs via apt, or fetches official Linux AVX2 binary on Colab."""
    for cand in [
        shutil.which("stockfish"),
        "/content/bin/stockfish",
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
    ]:
        if cand and Path(cand).exists():
            return Path(cand)

    # 1. Try apt-get on Colab Linux VM
    if Path("/content").exists() and shutil.which("apt-get"):
        try:
            subprocess.run(
                ["apt-get", "update", "-qq"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=45,
                check=False,
            )
            subprocess.run(
                ["apt-get", "install", "-qq", "-y", "stockfish"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
                check=False,
            )
            for cand in [shutil.which("stockfish"), "/usr/games/stockfish", "/usr/bin/stockfish"]:
                if cand and Path(cand).exists():
                    return Path(cand)
        except Exception:
            pass

    # 2. Direct fallback: download official Stockfish Linux x86-64 AVX2 release tarball (~2s!)
    if Path("/content").exists():
        try:
            import os
            import tarfile
            import urllib.request

            sf_dir = Path("/content/bin")
            sf_dir.mkdir(parents=True, exist_ok=True)
            tar_path = sf_dir / "sf.tar"
            url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar"
            req = urllib.request.Request(url, headers={"User-Agent": "Saccadion-Arena/1.0"})
            with urllib.request.urlopen(req, timeout=45) as resp, open(tar_path, "wb") as out:
                shutil.copyfileobj(resp, out)
            with tarfile.open(tar_path, "r") as tf:
                for member in tf.getmembers():
                    if member.isfile() and "stockfish-ubuntu" in Path(member.name).name:
                        src = tf.extractfile(member)
                        if src is not None:
                            dst = sf_dir / "stockfish"
                            with open(dst, "wb") as f_out:
                                shutil.copyfileobj(src, f_out)
                            os.chmod(dst, 0o755)
                            tar_path.unlink(missing_ok=True)
                            return dst
            tar_path.unlink(missing_ok=True)
        except Exception:
            pass
    return None


def _material_eval_cp(board: Any) -> int:
    """Fast material + center-control fallback evaluator for Baseline-Material-d2 (~900 Elo)."""
    import chess

    score = 0
    for pt, val in PIECE_VALUES_CP.items():
        if val == 0:
            continue
        score += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return score if board.turn == chess.WHITE else -score


def _baseline_d2_move(board: Any, rng: random.Random) -> Any:
    """2-ply alpha-beta material player (~900 Elo) for reliable Epoch-1 calibration."""
    moves = list(board.legal_moves)
    rng.shuffle(moves)
    best_mv = moves[0]
    best_sc = -999999
    for mv in moves[:30]:
        board.push(mv)
        if board.is_checkmate():
            board.pop()
            return mv
        # Opponent best 1-ply reply
        opp_best = -999999
        for omv in list(board.legal_moves)[:20]:
            board.push(omv)
            if board.is_checkmate():
                sc = 100000
            else:
                sc = -_material_eval_cp(board)
            board.pop()
            if sc > opp_best:
                opp_best = sc
        sc_me = -opp_best if opp_best > -999999 else _material_eval_cp(board)
        board.pop()
        if sc_me > best_sc:
            best_sc = sc_me
            best_mv = mv
    return best_mv


class _UciProcess:
    """Minimal, leak-free UCI subprocess wrapper."""

    def __init__(self, cmd: List[str], options: Optional[Dict[str, str]] = None):
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        if options:
            for k, v in options.items():
                self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, line: str) -> None:
        if self.proc.stdin:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()

    def _read_until(self, token: str, timeout_s: float = 10.0) -> List[str]:
        lines: List[str] = []
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.stdout is None:
                break
            line = self.proc.stdout.readline()
            if not line:
                break
            s = line.strip()
            lines.append(s)
            if s.startswith(token) or token in s:
                break
        return lines

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def set_option(self, name: str, value: str) -> None:
        self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._read_until("readyok")

    def go(self, fen: str, movetime_ms: int = 80) -> Tuple[Optional[str], Optional[int]]:
        self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime_ms}")
        lines = self._read_until("bestmove", timeout_s=max(5.0, movetime_ms / 1000.0 + 4.0))
        best_mv = None
        last_cp = None
        for ln in lines:
            if " score cp " in ln:
                try:
                    last_cp = int(ln.split(" score cp ")[1].split()[0])
                except Exception:
                    pass
            elif " score mate " in ln:
                try:
                    m = int(ln.split(" score mate ")[1].split()[0])
                    last_cp = 2500 if m > 0 else -2500
                except Exception:
                    pass
            if ln.startswith("bestmove "):
                toks = ln.split()
                if len(toks) >= 2 and toks[1] != "(none)":
                    best_mv = toks[1]
        return best_mv, last_cp

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=1.5)
        except Exception:
            try:
                self.proc.kill()
                self.proc.wait(timeout=2.0)
            except Exception:
                pass
        for stream in (self.proc.stdin, self.proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


@dataclass
class ArenaStatus:
    active: bool = False
    engine_mode: str = "waiting_weights"
    stockfish_available: bool = False
    opponent_name: str = "Baseline-Material-d2"
    opponent_elo: int = 900
    level_idx: int = 0
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    real_elo: float = 0.0
    elo_ci: float = 999.0
    harvested_positions: int = 0
    current_game_ply: int = 0
    last_result: str = "Warming up..."
    history_elo: List[Tuple[int, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "engine_mode": self.engine_mode,
            "stockfish_available": self.stockfish_available,
            "opponent_name": self.opponent_name,
            "opponent_elo": self.opponent_elo,
            "level_idx": self.level_idx,
            "games_played": self.games_played,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "real_elo": self.real_elo,
            "elo_ci": self.elo_ci,
            "harvested_positions": self.harvested_positions,
            "current_game_ply": self.current_game_ply,
            "last_result": self.last_result,
        }


class StockfishSparringArena:
    """Runs asynchronous background CPU sparring games while the TPU trains."""

    def __init__(
        self,
        weights_path: Path,
        harvest_shard_path: Path,
        status_json_path: Path,
        rust_uci_binary: Optional[Path] = None,
        movetime_ms: int = 60,
        max_plies: int = 120,
    ):
        self.weights_path = weights_path
        self.harvest_shard_path = harvest_shard_path
        self.status_json_path = status_json_path
        self.rust_uci_binary = rust_uci_binary
        self.movetime_ms = movetime_ms
        self.max_plies = max_plies

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status = ArenaStatus()
        self._results: List[Tuple[int, float]] = []
        self._recent_level_scores: List[float] = []
        self._harvest_records: List[np.ndarray] = []
        self._last_weights_mtime: float = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run_loop, name="SaccadionCPUArena", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return self._status.to_dict()

    def _find_rust_uci(self) -> Optional[Path]:
        if self.rust_uci_binary and self.rust_uci_binary.exists():
            return self.rust_uci_binary
        repo_root = Path(__file__).resolve().parents[2]
        for cand in [
            repo_root / "target" / "release" / "saccadion",
            repo_root / "target" / "release" / "saccadion.exe",
            Path("/content/drive/MyDrive/Saccadion_TPU/bin_v8/saccadion"),
            Path("/content/drive/MyDrive/Saccadion_TPU/bin_v7/saccadion"),
            Path("/content/drive/MyDrive/Saccadion_TPU/bin_v6/saccadion"),
            Path("/content/drive/MyDrive/Saccadion_TPU/bin_v5/saccadion"),
        ]:
            if cand.exists():
                return cand
        return None

    def _pack_harvest_record(self, board: Any, best_uci: str, cp_stm: int) -> Optional[np.ndarray]:
        import chess

        try:
            mv = chess.Move.from_uci(best_uci)
            if mv not in board.legal_moves:
                return None
        except Exception:
            return None

        piece_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }
        rec = np.zeros((), dtype=SACC_RECORD_DTYPE)
        atk_w, atk_b = chess.BB_EMPTY, chess.BB_EMPTY
        for sq, pc in board.piece_map().items():
            code = (1 if pc.color == chess.WHITE else 7) + piece_map[pc.piece_type]
            if (sq & 1) == 0:
                rec["board_nibbles"][sq >> 1] |= code
            else:
                rec["board_nibbles"][sq >> 1] |= code << 4
            atks = board.attacks_mask(sq)
            if pc.color == chess.WHITE:
                atk_w |= atks
            else:
                atk_b |= atks

        stm = 0 if board.turn == chess.WHITE else 1
        castling = (
            (1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
            | (2 if board.has_queenside_castling_rights(chess.WHITE) else 0)
            | (4 if board.has_kingside_castling_rights(chess.BLACK) else 0)
            | (8 if board.has_queenside_castling_rights(chess.BLACK) else 0)
        )
        ep_f = (chess.square_file(board.ep_square) + 1) if board.ep_square is not None else 0
        phase = min(7, max(0, (len(board.piece_map()) - 2) // 4))
        rec["meta_flags"] = (
            stm | (castling << 1) | (ep_f << 5) | (min(255, board.halfmove_clock) << 9) | (phase << 17)
        )
        w = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (cp_stm - 90.0) / 140.0))))
        l = 1.0 / (1.0 + math.exp(max(-20.0, min(20.0, (cp_stm + 90.0) / 140.0))))
        d = max(0.02, 1.0 - w - l)
        s = w + d + l
        rec["wdl_target"] = [w / s, d / s, l / s]

        bm = (mv.from_square << 6) | mv.to_square
        rec["best_move_u16"] = bm
        rec["next_move_u16"] = bm
        rec["top4_moves"] = [bm, 0, 0, 0]
        rec["top4_probs_u16"] = [65535, 0, 0, 0]
        rec["pressure_targets"] = [0.25, 0.25, 0.25, 0.15, 0.35]
        rec["lh_targets"] = [0.5, 0.0, 0.5 if board.is_check() else 0.0, 0.0]
        rec["volatility"] = 0.80
        rec["mlh"] = 0.45

        def orient_bb(bb: int) -> int:
            return bb if stm == 0 else int.from_bytes(bb.to_bytes(8, "little")[::-1], "little")

        rec["threat_mask_stm"] = orient_bb(atk_w if stm == 0 else atk_b)
        rec["threat_mask_opp"] = orient_bb(atk_b if stm == 0 else atk_w)
        return rec

    def _flush_harvest(self) -> None:
        if not self._harvest_records:
            return
        self.harvest_shard_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(self._harvest_records[-20000:], dtype=SACC_RECORD_DTYPE)
        tmp = self.harvest_shard_path.with_suffix(".sacc.tmp")
        with open(tmp, "wb") as f:
            f.write(struct.pack("<8sII", SACC_SHARD_MAGIC, 0, len(arr)))
            f.write(arr.tobytes())
        tmp.replace(self.harvest_shard_path)

    def _save_status(self) -> None:
        try:
            self.status_json_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_json_path.write_text(json.dumps(self.get_status(), indent=2), encoding="utf-8")
        except Exception:
            pass

    def _run_loop(self) -> None:
        try:
            import chess
        except ImportError:
            with self._lock:
                self._status.last_result = "python-chess not installed"
            return

        sf_bin = find_or_install_stockfish()
        rust_bin = self._find_rust_uci()
        with self._lock:
            self._status.stockfish_available = sf_bin is not None
            self._status.engine_mode = "Pure-Rust UCI" if rust_bin else "Waiting Rust UCI"
            self._status.active = True

        # Wait gently until at least one `.saccnet` checkpoint exists
        while not self._stop_evt.is_set() and not self.weights_path.exists():
            with self._lock:
                self._status.last_result = "Waiting for Epoch-1 .saccnet export..."
            time.sleep(2.0)

        if self._stop_evt.is_set():
            return

        if rust_bin is None:
            with self._lock:
                self._status.engine_mode = "No Rust binary"
                self._status.last_result = "Build saccadion-uci to enable live CPU sparring"
            return

        sacc_proc: Optional[_UciProcess] = None
        sf_proc: Optional[_UciProcess] = None
        rng = random.Random(2026)

        # Diverse opening suite (6-ply book) so sparring games never repeat the same game
        opening_lines = [
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
            ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"],
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"],
            ["c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5"],
            ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],
            ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"],
            ["g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "c7c6"],
            ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"],
        ]

        try:
            if sf_bin is not None:
                sf_proc = _UciProcess([str(sf_bin)], {"Threads": "1", "Hash": "16"})

            level_idx = 1 if sf_proc is not None else 0
            game_num = 0

            while not self._stop_evt.is_set():
                # Hot-reload Saccadion UCI whenever a newer `.saccnet` is exported by TPU!
                mtime = self.weights_path.stat().st_mtime if self.weights_path.exists() else 0.0
                if sacc_proc is None or mtime > self._last_weights_mtime:
                    if sacc_proc is not None:
                        sacc_proc.close()
                    sacc_proc = _UciProcess([str(rust_bin)], {"Hash": "32", "Threads": "2"})
                    sacc_proc.set_option("WeightsPath", str(self.weights_path))
                    self._last_weights_mtime = mtime

                opp_name, opp_elo, sf_elo = OPPONENT_LADDER[level_idx]
                if sf_elo is not None and sf_proc is None:
                    level_idx = 0
                    opp_name, opp_elo, sf_elo = OPPONENT_LADDER[0]

                if sf_proc is not None and sf_elo is not None:
                    sf_proc.set_option("UCI_LimitStrength", "true")
                    sf_proc.set_option("UCI_Elo", str(min(3190, max(1320, sf_elo))))
                    sf_proc.new_game()
                sacc_proc.new_game()

                board = chess.Board()
                for u in rng.choice(opening_lines):
                    mv = chess.Move.from_uci(u)
                    if mv in board.legal_moves:
                        board.push(mv)

                sacc_is_white = (game_num % 2) == 0
                game_num += 1
                sacc_positions: List[Tuple[str, int]] = []

                with self._lock:
                    self._status.opponent_name = opp_name
                    self._status.opponent_elo = opp_elo
                    self._status.level_idx = level_idx
                    self._status.current_game_ply = 0

                for ply in range(self.max_plies):
                    if self._stop_evt.is_set() or board.is_game_over():
                        break
                    with self._lock:
                        self._status.current_game_ply = ply + 1

                    is_sacc_turn = (board.turn == chess.WHITE) == sacc_is_white
                    if is_sacc_turn:
                        mv_u, cp = sacc_proc.go(board.fen(), movetime_ms=self.movetime_ms)
                        sacc_positions.append((board.fen(), cp or 0))
                    else:
                        if sf_elo is not None and sf_proc is not None:
                            mv_u, _ = sf_proc.go(board.fen(), movetime_ms=max(25, self.movetime_ms // 2))
                        else:
                            mv_u = _baseline_d2_move(board, rng).uci()

                    if not mv_u:
                        break
                    try:
                        mv = chess.Move.from_uci(mv_u)
                        if mv not in board.legal_moves:
                            break
                        board.push(mv)
                    except Exception:
                        break

                # Adjudicate game result
                outcome = board.outcome()
                if outcome is not None:
                    if outcome.winner is None:
                        score = 0.5
                    elif (outcome.winner == chess.WHITE) == sacc_is_white:
                        score = 1.0
                    else:
                        score = 0.0
                else:
                    # Clean material adjudication on max-ply cutoff
                    w_mat = _material_eval_cp(board) * (1 if board.turn == chess.WHITE else -1)
                    sacc_mat = w_mat if sacc_is_white else -w_mat
                    score = 1.0 if sacc_mat >= 220 else (0.0 if sacc_mat <= -220 else 0.5)

                # Harvest anti-Stockfish hard positions when Saccadion loses or draws a tough game
                if score <= 0.5 and sf_proc is not None and sacc_positions:
                    sf_proc.set_option("UCI_LimitStrength", "false")
                    sample_fens = rng.sample(sacc_positions, min(4, len(sacc_positions)))
                    for fen_s, _ in sample_fens:
                        if self._stop_evt.is_set():
                            break
                        b_h = chess.Board(fen_s)
                        best_ref, ref_cp = sf_proc.go(fen_s, movetime_ms=35)
                        if best_ref:
                            rec = self._pack_harvest_record(b_h, best_ref, ref_cp or 0)
                            if rec is not None:
                                self._harvest_records.append(rec)
                    self._flush_harvest()

                self._results.append((opp_elo, score))
                self._recent_level_scores.append(score)
                real_elo, elo_ci = estimate_mle_elo(self._results[-60:])

                # Adaptive opponent ladder promotion/demotion
                if len(self._recent_level_scores) >= 4:
                    avg = sum(self._recent_level_scores[-4:]) / 4.0
                    max_lv = len(OPPONENT_LADDER) - 1 if sf_proc is not None else 0
                    if avg >= 0.65 and level_idx < max_lv:
                        level_idx += 1
                        self._recent_level_scores.clear()
                    elif avg <= 0.20 and level_idx > 0:
                        level_idx -= 1
                        self._recent_level_scores.clear()

                res_txt = "WIN" if score == 1.0 else ("LOSS" if score == 0.0 else "DRAW")
                with self._lock:
                    self._status.games_played += 1
                    if score == 1.0:
                        self._status.wins += 1
                    elif score == 0.0:
                        self._status.losses += 1
                    else:
                        self._status.draws += 1
                    self._status.real_elo = real_elo
                    self._status.elo_ci = elo_ci
                    self._status.harvested_positions = len(self._harvest_records)
                    self._status.last_result = f"G{game_num}: {res_txt} vs {opp_name} ({len(board.move_stack)} plies)"
                self._save_status()
                # Yield politely so host OS stays responsive for TPU feeding
                time.sleep(0.15)
        finally:
            if sacc_proc is not None:
                sacc_proc.close()
            if sf_proc is not None:
                sf_proc.close()
            with self._lock:
                self._status.active = False
            self._save_status()
