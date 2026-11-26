"""Interactive Terminal Match vs Pure-Rust Saccadion UCI Engine.

Usage:
    python training/play_vs_saccadion.py --weights saccadion_best.saccnet --side white --movetime 500
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import chess


def _find_saccadion_bin(repo_root: Path) -> Path:
    for cand in [
        repo_root / "target" / "release" / "saccadion.exe",
        repo_root / "target" / "release" / "saccadion",
    ]:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Pure-Rust `saccadion` binary not found. Build it first with:\n"
        "  cargo build --release -p saccadion-uci"
    )


def _send(proc: subprocess.Popen, cmd: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def _read_until(proc: subprocess.Popen, token: str) -> list[str]:
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        s = line.strip()
        lines.append(s)
        if s.startswith(token) or token in s:
            break
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Play chess in terminal against Pure-Rust Saccadion")
    parser.add_argument("--weights", required=True, help="Path to trained `.saccnet` weights file")
    parser.add_argument("--side", choices=["white", "black"], default="white", help="Your color")
    parser.add_argument("--movetime", type=int, default=500, help="Saccadion think time per move (ms)")
    parser.add_argument("--hash", type=int, default=64, help="Transposition table size (MB)")
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"Error: weights file not found: {weights_path}")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[1]
    exe = _find_saccadion_bin(repo_root)

    proc = subprocess.Popen(
        [str(exe)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        _send(proc, "uci")
        _read_until(proc, "uciok")
        _send(proc, f"setoption name Hash value {args.hash}")
        _send(proc, f"setoption name WeightsPath value {weights_path}")
        _send(proc, "isready")
        info_lines = _read_until(proc, "readyok")
        for ln in info_lines:
            if ln.startswith("info string"):
                print(ln)
        _send(proc, "ucinewgame")
        _send(proc, "isready")
        _read_until(proc, "readyok")

        board = chess.Board()
        human_is_white = args.side == "white"
        print(f"\n=== MATCH START: YOU ({args.side.upper()}) vs SACCADION ({args.movetime} ms/move) ===")
        print("Enter moves in UCI (e.g. e2e4) or SAN (e.g. Nf3, O-O). Type 'quit' to exit.\n")

        while not board.is_game_over():
            print(board.unicode(borders=True) if hasattr(board, "unicode") else board)
            print(f"FEN: {board.fen()}\n")

            if (board.turn == chess.WHITE) == human_is_white:
                while True:
                    raw = input("Your move > ").strip()
                    if raw.lower() in ("quit", "exit", "q"):
                        return
                    mv = None
                    try:
                        mv = board.parse_san(raw)
                    except Exception:
                        try:
                            cand = chess.Move.from_uci(raw)
                            if cand in board.legal_moves:
                                mv = cand
                        except Exception:
                            mv = None
                    if mv is not None and mv in board.legal_moves:
                        board.push(mv)
                        break
                    print("Illegal move! Try e.g. e2e4, Nf3, or O-O.")
            else:
                _send(proc, f"position fen {board.fen()}")
                _send(proc, f"go movetime {args.movetime}")
                lines = _read_until(proc, "bestmove")
                last_info = next((ln for ln in reversed(lines) if ln.startswith("info depth")), "")
                bm_line = lines[-1] if lines else ""
                toks = bm_line.split()
                if len(toks) < 2 or toks[1] == "(none)":
                    print("Saccadion has no legal moves.")
                    break
                mv = chess.Move.from_uci(toks[1])
                san = board.san(mv)
                board.push(mv)
                print(f"Saccadion plays: {san} ({toks[1]})")
                if last_info:
                    print(f"  [{last_info}]\n")

        print(board)
        print("Game over! Result:", board.result(), "|", board.outcome())
    finally:
        try:
            _send(proc, "quit")
            proc.wait(timeout=2.0)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
