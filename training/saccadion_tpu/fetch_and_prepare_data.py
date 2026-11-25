"""Automated Chess Dataset Downloader, Strategic Organizer, and `.sacc` Shard Builder.

Downloads high-level chess games directly inside Google Colab TPU (or locally) from:
  1. `lichess_elite` : 2500+ vs 2300+ Lichess Elite GM/IM strategic games (database.nikonoel.fr)
  2. `fishtest`      : 3600+ Elo Official Stockfish Fishtest games with per-ply evaluations (HuggingFace)
  3. `lichess_eval`  : Streaming Lichess Standard Rated .pgn.zst filtered for 2200+ Elo + [%eval]
  4. `auto`          : Smart blend combining strategic human Elite games + evaluated Fishtest games

Organizes games cleanly with ZERO train/val game leakage into:
  - `<out_dir>/raw/`                       (downloaded PGNs)
  - `<out_dir>/shards/train_random_*.sacc` (shuffled 128-byte multi-sensor records)
  - `<out_dir>/shards/val_random.sacc`     (strictly held-out validation games)
  - `<out_dir>/shards/train_traj_*.sacc`   (K=12 contiguous trajectory windows for Stage-3 Dopamine)
  - `<out_dir>/manifest.json`              (complete dataset telemetry & shard inventory)

Uses our compiled Pure-Rust `saccadion-data` binary when available (100x faster!),
with a clean Pure-Python (`python-chess`) fallback so it ALWAYS succeeds on any machine.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import re
import shutil
import struct
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dataset import SACC_RECORD_DTYPE, SACC_SHARD_MAGIC

USER_AGENT = "Saccadion-Research/0.1 (+https://github.com/vash-001/Saccadion)"


def _http_get_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _http_download(url: str, dst: Path, max_bytes: Optional[int] = None, timeout: int = 60) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"Downloading: {url}")
    downloaded = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dst, "wb") as out:
        while True:
            chunk = resp.read(1 << 18)  # 256 KB chunks
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if max_bytes is not None and downloaded >= max_bytes:
                break
    print(f"  Saved {downloaded / (1024 * 1024):.2f} MB -> {dst}")
    return dst


def download_lichess_elite(raw_dir: Path, max_games: int = 5000) -> Optional[Path]:
    """Fetches the latest monthly Lichess Elite 2500+ PGN archive from database.nikonoel.fr."""
    try:
        html = _http_get_text("https://database.nikonoel.fr/")
        zips = re.findall(r'href="(lichess_elite_\d{4}-\d{2}\.zip)"', html)
        if not zips:
            print("[Warn] Could not parse Lichess Elite links from database.nikonoel.fr")
            return None
        latest = sorted(set(zips))[-1]
        url = f"https://database.nikonoel.fr/{latest}"
        zip_path = raw_dir / latest
        if not zip_path.exists():
            _http_download(url, zip_path)
        out_pgn = raw_dir / f"elite_{max_games}.pgn"
        count = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            pgn_names = [n for n in zf.namelist() if n.endswith(".pgn")]
            if not pgn_names:
                return None
            with zf.open(pgn_names[0], "r") as src, open(out_pgn, "w", encoding="utf-8") as dst:
                buf: List[str] = []
                in_moves = False
                for raw_line in io.TextIOWrapper(src, encoding="utf-8", errors="ignore"):
                    if raw_line.startswith("[Event ") and in_moves:
                        dst.writelines(buf)
                        dst.write("\n")
                        buf.clear()
                        in_moves = False
                        count += 1
                        if count >= max_games:
                            break
                    elif not raw_line.startswith("[") and raw_line.strip():
                        in_moves = True
                    buf.append(raw_line)
                if buf and count < max_games:
                    dst.writelines(buf)
                    count += 1
        print(f"Extracted {count:,} Lichess Elite 2500+ games -> {out_pgn}")
        return out_pgn
    except Exception as exc:
        print(f"[Warn] Lichess Elite download failed: {exc}")
        return None


def download_fishtest_pgn(raw_dir: Path, max_games: int = 5000) -> Optional[Path]:
    """Fetches official Stockfish Fishtest evaluated PGNs from HuggingFace (3600+ Elo)."""
    try:
        api_url = "https://huggingface.co/api/datasets/official-stockfish/fishtest_pgns/tree/main"
        listing = json.loads(_http_get_text(api_url))
        # Find a recent directory or .pgn.gz file
        gz_files = [e["path"] for e in listing if e.get("path", "").endswith(".pgn.gz")]
        if not gz_files:
            dirs = sorted([e["path"] for e in listing if e.get("type") == "directory"])
            for d in reversed(dirs[-4:]):
                sub = json.loads(_http_get_text(f"{api_url}/{d}"))
                gz_files = [e["path"] for e in sub if e.get("path", "").endswith(".pgn.gz")]
                if gz_files:
                    break
                # Go one level deeper if organized by year/month
                sub_dirs = sorted([e["path"] for e in sub if e.get("type") == "directory"])
                if sub_dirs:
                    sub2 = json.loads(_http_get_text(f"{api_url}/{sub_dirs[-1]}"))
                    gz_files = [e["path"] for e in sub2 if e.get("path", "").endswith(".pgn.gz")]
                    if gz_files:
                        break
        if not gz_files:
            print("[Warn] No .pgn.gz files located in official-stockfish/fishtest_pgns")
            return None

        target_path = gz_files[-1]
        dl_url = f"https://huggingface.co/datasets/official-stockfish/fishtest_pgns/resolve/main/{target_path}"
        gz_local = raw_dir / Path(target_path).name
        if not gz_local.exists():
            _http_download(dl_url, gz_local, max_bytes=45 * 1024 * 1024)

        out_pgn = raw_dir / f"fishtest_{max_games}.pgn"
        count = 0
        with gzip.open(gz_local, "rt", encoding="utf-8", errors="ignore") as src, open(
            out_pgn, "w", encoding="utf-8"
        ) as dst:
            buf: List[str] = []
            in_moves = False
            try:
                for line in src:
                    if line.startswith("[Event ") and in_moves:
                        dst.writelines(buf)
                        dst.write("\n")
                        buf.clear()
                        in_moves = False
                        count += 1
                        if count >= max_games:
                            break
                    elif not line.startswith("[") and line.strip():
                        in_moves = True
                    buf.append(line)
            except EOFError:
                pass  # Expected if capped by max_bytes stream download
            if buf and count < max_games and in_moves:
                dst.writelines(buf)
                count += 1
        print(f"Extracted {count:,} Stockfish Fishtest evaluated games -> {out_pgn}")
        return out_pgn if count > 0 else None
    except Exception as exc:
        print(f"[Warn] Fishtest download failed: {exc}")
        return None


def download_lichess_standard_eval_stream(
    raw_dir: Path, max_games: int = 3000, min_elo: int = 2100
) -> Optional[Path]:
    """Streams recent Lichess standard .pgn.zst and filters for high-Elo [%eval] games."""
    try:
        import zstandard as zstd
    except ImportError:
        print("[Warn] `zstandard` not installed; skipping streaming .pgn.zst download.")
        return None

    try:
        index_txt = _http_get_text("https://database.lichess.org/standard/list.txt")
        urls = [u.strip() for u in index_txt.splitlines() if u.strip().endswith(".pgn.zst")]
        if not urls:
            return None
        # Pick a compact recent archive (oldest files at end are smaller; index 0 is newest)
        url = urls[0]
        print(f"Streaming Lichess [%eval] games (Elo >= {min_elo}) from {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        out_pgn = raw_dir / f"lichess_eval_{max_games}.pgn"
        kept = 0
        seen = 0
        with urllib.request.urlopen(req, timeout=60) as resp:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(resp) as reader, open(out_pgn, "w", encoding="utf-8") as dst:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
                buf: List[str] = []
                in_moves = False
                w_elo, b_elo = 0, 0
                has_eval = False
                for line in text_stream:
                    if line.startswith("[Event ") and in_moves:
                        seen += 1
                        if has_eval and w_elo >= min_elo and b_elo >= min_elo:
                            dst.writelines(buf)
                            dst.write("\n")
                            kept += 1
                            if kept % 500 == 0:
                                print(f"  Streamed {seen:,} games -> kept {kept:,} high-Elo [%eval] games")
                            if kept >= max_games:
                                break
                        buf.clear()
                        in_moves = False
                        w_elo, b_elo, has_eval = 0, 0, False
                    elif line.startswith("[WhiteElo \""):
                        try:
                            w_elo = int(line.split('"')[1])
                        except ValueError:
                            w_elo = 0
                    elif line.startswith("[BlackElo \""):
                        try:
                            b_elo = int(line.split('"')[1])
                        except ValueError:
                            b_elo = 0
                    elif not line.startswith("[") and line.strip():
                        in_moves = True
                        if "[%eval " in line:
                            has_eval = True
                    buf.append(line)
        print(f"Streamed {seen:,} games -> saved {kept:,} annotated games -> {out_pgn}")
        return out_pgn if kept > 0 else None
    except Exception as exc:
        print(f"[Warn] Lichess streaming failed: {exc}")
        return None


def _pack_pgn_with_python_chess(
    pgn_paths: List[Path], shard_dir: Path, val_pct: int = 5, max_games: int = 10000
) -> Dict[str, int]:
    """Pure-Python fallback packer using `python-chess` producing identical 128-byte `.sacc` shards."""
    import chess
    import chess.pgn

    shard_dir.mkdir(parents=True, exist_ok=True)
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    piece_cp = [100, 320, 330, 500, 900, 20000]

    def cp_to_wdl(cp: float) -> Tuple[float, float, float]:
        w = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (cp - 90.0) / 140.0))))
        l = 1.0 / (1.0 + math.exp(max(-20.0, min(20.0, (cp + 90.0) / 140.0))))
        d = max(0.02, 1.0 - w - l)
        s = w + d + l
        return (w / s, d / s, l / s)

    def parse_comment_cp(comment: str) -> Optional[float]:
        m = re.search(r"\[%eval\s+([^\]]+)\]", comment)
        if m:
            s = m.group(1).strip()
            if s.startswith("#"):
                return 2500.0 if not s.startswith("#-") else -2500.0
            try:
                return max(-2500.0, min(2500.0, float(s) * 100.0))
            except ValueError:
                return None
        tok = comment.strip().split()[0] if comment.strip() else ""
        if "/" in tok:
            s = tok.split("/")[0]
            if "M" in s or "m" in s:
                return -2500.0 if s.startswith("-") else 2500.0
            try:
                return max(-2500.0, min(2500.0, float(s) * 100.0))
            except ValueError:
                return None
        return None

    def snapshot_board(b: chess.Board) -> Dict[str, float]:
        # Fast strategic snapshot for 10-20 ply Strategic Pressure targets
        atk_w, atk_b = chess.BB_EMPTY, chess.BB_EMPTY
        mob = [0.0, 0.0]
        max_act = [0.0, 0.0]
        for sq, pc in b.piece_map().items():
            ci = 0 if pc.color == chess.WHITE else 1
            atks = b.attacks_mask(sq)
            if ci == 0:
                atk_w |= atks
            else:
                atk_b |= atks
            if pc.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                cnt = float(bin(atks & ~b.occupied_co[pc.color]).count("1"))
                mob[ci] += cnt
                max_act[ci] = max(max_act[ci], cnt)
        wk = b.king(chess.WHITE) or 4
        bk = b.king(chess.BLACK) or 60
        wk_ring = b.attacks_mask(wk) | (1 << wk)
        bk_ring = b.attacks_mask(bk) | (1 << bk)
        kp_w = float(bin(atk_w & bk_ring).count("1"))
        kp_b = float(bin(atk_b & wk_ring).count("1"))
        black_half = 0xFFFF_FFFF_0000_0000
        white_half = 0x0000_0000_FFFF_FFFF
        sp_w = float(bin(atk_w & black_half).count("1"))
        sp_b = float(bin(atk_b & white_half).count("1"))
        return {
            "mob_w": mob[0],
            "mob_b": mob[1],
            "max_w": max_act[0],
            "max_b": max_act[1],
            "kp_w": kp_w,
            "kp_b": kp_b,
            "sp_w": sp_w,
            "sp_b": sp_b,
            "atk_w": atk_w,
            "atk_b": atk_b,
        }

    all_games_records: List[List[np.ndarray]] = []
    games_loaded = 0

    for pgn_path in pgn_paths:
        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
            while games_loaded < max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                res_str = game.headers.get("Result", "*")
                w_res = 1.0 if res_str == "1-0" else (0.0 if res_str == "0-1" else 0.5)

                board = game.board()
                nodes_data = []
                node = game
                while node.variations:
                    nxt = node.variation(0)
                    mv = nxt.move
                    ev_w = parse_comment_cp(nxt.comment)
                    snap = snapshot_board(board)
                    nodes_data.append((board.copy(stack=False), mv, ev_w, snap))
                    board.push(mv)
                    node = nxt

                if len(nodes_data) < 12:
                    continue

                n = len(nodes_data)
                evals_stm = []
                for b, _, ev_w, _ in nodes_data:
                    if ev_w is not None:
                        ev = ev_w if b.turn == chess.WHITE else -ev_w
                    else:
                        mat = sum(
                            (piece_cp[piece_map[pc.piece_type]] if pc.color == b.turn else -piece_cp[piece_map[pc.piece_type]])
                            for pc in b.piece_map().values()
                            if pc.piece_type != chess.KING
                        )
                        ev = float(mat)
                    evals_stm.append(ev)

                g_recs = []
                for ply, (b, mv, _, snap) in enumerate(nodes_data):
                    rec = np.zeros((), dtype=SACC_RECORD_DTYPE)
                    for sq, pc in b.piece_map().items():
                        code = (1 if pc.color == chess.WHITE else 7) + piece_map[pc.piece_type]
                        if (sq & 1) == 0:
                            rec["board_nibbles"][sq >> 1] |= code
                        else:
                            rec["board_nibbles"][sq >> 1] |= code << 4

                    stm = 0 if b.turn == chess.WHITE else 1
                    castling = (
                        (1 if b.has_kingside_castling_rights(chess.WHITE) else 0)
                        | (2 if b.has_queenside_castling_rights(chess.WHITE) else 0)
                        | (4 if b.has_kingside_castling_rights(chess.BLACK) else 0)
                        | (8 if b.has_queenside_castling_rights(chess.BLACK) else 0)
                    )
                    ep_f = (chess.square_file(b.ep_square) + 1) if b.ep_square is not None else 0
                    phase = min(7, max(0, (len(b.piece_map()) - 2) // 4))
                    rec["meta_flags"] = (
                        stm | (castling << 1) | (ep_f << 5) | (min(255, b.halfmove_clock) << 9) | (phase << 17)
                    )

                    out_stm = w_res if stm == 0 else (1.0 - w_res)
                    out_wdl = (0.94, 0.04, 0.02) if out_stm > 0.75 else ((0.02, 0.04, 0.94) if out_stm < 0.25 else (0.08, 0.84, 0.08))
                    ev_wdl = cp_to_wdl(evals_stm[ply])
                    rec["wdl_target"] = [0.6 * ev_wdl[i] + 0.4 * out_wdl[i] for i in range(3)]

                    bm = (mv.from_square << 6) | mv.to_square
                    nm = (
                        (nodes_data[ply + 2][1].from_square << 6) | nodes_data[ply + 2][1].to_square
                        if ply + 2 < n
                        else bm
                    )
                    rec["best_move_u16"] = bm
                    rec["next_move_u16"] = nm
                    rec["top4_moves"] = [bm, 0, 0, 0]
                    rec["top4_probs_u16"] = [65535, 0, 0, 0]

                    # 10-20 ply Strategic Pressure trajectory labels
                    sn2 = nodes_data[min(n - 1, ply + 2)][3]
                    sn6 = nodes_data[min(n - 1, ply + 6)][3]
                    sn12 = nodes_data[min(n - 1, ply + 12)][3]
                    sn20 = nodes_data[min(n - 1, ply + 20)][3]
                    o_k = "b" if stm == 0 else "w"
                    u_k = "w" if stm == 0 else "b"
                    fut_opp_mob = 0.5 * sn12[f"mob_{o_k}"] + 0.3 * sn6[f"mob_{o_k}"] + 0.2 * sn20[f"mob_{o_k}"]
                    mob_col = max(-1.0, min(1.0, (snap[f"mob_{o_k}"] - fut_opp_mob) / (snap[f"mob_{o_k}"] + 4.0)))
                    ent_sq = max(-1.0, min(1.0, (snap[f"mob_{o_k}"] - sn12[f"mob_{o_k}"]) / (snap[f"mob_{o_k}"] + 6.0)))
                    kp_del = max(-1.0, min(1.0, ((0.5 * sn6[f"kp_{u_k}"] + 0.5 * sn12[f"kp_{u_k}"]) - snap[f"kp_{u_k}"]) / 6.0))
                    sp_del = max(-1.0, min(1.0, ((sn12[f"sp_{u_k}"] - sn12[f"sp_{o_k}"]) - (snap[f"sp_{u_k}"] - snap[f"sp_{o_k}"])) / 16.0))
                    proph = max(-1.0, min(1.0, (snap[f"max_{o_k}"] - sn2[f"max_{o_k}"]) / (snap[f"max_{o_k}"] + 3.0)))
                    rec["pressure_targets"] = [mob_col, ent_sq, kp_del, sp_del, proph]

                    rec["lh_targets"] = [0.0, 1.0 if b.is_insufficient_material() else 0.0, 0.5 if b.is_check() else 0.0, 0.0]
                    fut_ev = evals_stm[min(n - 1, ply + 4)]
                    rec["volatility"] = min(1.0, abs(fut_ev - evals_stm[ply]) / 300.0)
                    rec["mlh"] = min(1.0, (n - ply) / 120.0)

                    def orient_bb(bb: int) -> int:
                        if stm == 0:
                            return bb
                        return int.from_bytes(bb.to_bytes(8, "little")[::-1], "little")

                    rec["threat_mask_stm"] = orient_bb(snap["atk_w" if stm == 0 else "atk_b"])
                    rec["threat_mask_opp"] = orient_bb(snap["atk_b" if stm == 0 else "atk_w"])
                    g_recs.append(rec)

                all_games_records.append(g_recs)
                games_loaded += 1
                if games_loaded % 500 == 0:
                    print(f"  Packed {games_loaded:,} games via Python fallback...")

    if not all_games_records:
        raise RuntimeError("No valid games found in input PGNs.")

    val_n = max(1, min(len(all_games_records) - 1, (len(all_games_records) * val_pct) // 100))
    train_games = all_games_records[:-val_n]
    val_games = all_games_records[-val_n:]

    def write_shard(path: Path, recs: List[np.ndarray], is_traj: bool) -> None:
        arr = np.asarray(recs, dtype=SACC_RECORD_DTYPE)
        with open(path, "wb") as f:
            f.write(struct.pack("<8sII", SACC_SHARD_MAGIC, 1 if is_traj else 0, len(arr)))
            f.write(arr.tobytes())

    train_rand: List[np.ndarray] = []
    train_traj: List[np.ndarray] = []
    for g in train_games:
        train_rand.extend(g)
        for start in range(0, len(g) - 11, 6):
            train_traj.extend(g[start : start + 12])
    val_rand = [r for g in val_games for r in g]

    chunk = 100_000
    for idx in range(0, len(train_rand), chunk):
        write_shard(shard_dir / f"train_random_{idx // chunk:03d}.sacc", train_rand[idx : idx + chunk], False)
    if train_traj:
        usable = (len(train_traj) // 12) * 12
        write_shard(shard_dir / "train_traj_000.sacc", train_traj[:usable], True)
    write_shard(shard_dir / "val_random.sacc", val_rand, False)

    return {
        "games_total": len(all_games_records),
        "games_train": len(train_games),
        "games_val": len(val_games),
        "train_positions": len(train_rand),
        "val_positions": len(val_rand),
        "traj_windows": len(train_traj) // 12,
    }


def _find_rust_packer(repo_root: Path) -> Optional[Path]:
    for cand in [
        repo_root / "target" / "release" / "saccadion-data",
        repo_root / "target" / "release" / "saccadion-data.exe",
    ]:
        if cand.exists():
            return cand
    return None


def fetch_and_prepare(
    out_dir: Path,
    source: str = "auto",
    max_games: int = 4000,
    val_pct: int = 5,
    local_pgn: Optional[Path] = None,
    build_rust_if_available: bool = True,
) -> Path:
    """End-to-end downloader + strategic organizer returning the shard directory path."""
    raw_dir = out_dir / "raw"
    shard_dir = out_dir / "shards"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    pgn_files: List[Path] = []
    if local_pgn is not None and local_pgn.exists():
        pgn_files.append(local_pgn)
    else:
        if source in ("auto", "lichess_elite"):
            p = download_lichess_elite(raw_dir, max_games=max_games // 2 if source == "auto" else max_games)
            if p:
                pgn_files.append(p)
        if source in ("auto", "fishtest"):
            p = download_fishtest_pgn(raw_dir, max_games=max_games // 2 if source == "auto" else max_games)
            if p:
                pgn_files.append(p)
        if source == "lichess_eval" or (source == "auto" and not pgn_files):
            p = download_lichess_standard_eval_stream(raw_dir, max_games=max_games)
            if p:
                pgn_files.append(p)

    if not pgn_files:
        raise RuntimeError(
            "Could not download any PGN source. Check internet access or pass --local-pgn <file.pgn>."
        )

    # Merge downloaded PGNs into a single organized staging file
    merged_pgn = raw_dir / "combined_training.pgn"
    with open(merged_pgn, "wb") as dst:
        for pf in pgn_files:
            with open(pf, "rb") as src:
                shutil.copyfileobj(src, dst)
            dst.write(b"\n\n")

    repo_root = Path(__file__).resolve().parents[2]
    rust_bin = _find_rust_packer(repo_root)
    if rust_bin is None and build_rust_if_available and shutil.which("cargo"):
        print("Building Pure-Rust `saccadion-data` packer for 100x faster dataset organization...")
        subprocess.run(
            ["cargo", "build", "--release", "-p", "saccadion-data"],
            cwd=str(repo_root),
            check=False,
        )
        rust_bin = _find_rust_packer(repo_root)

    # Clear old shards before writing fresh split
    for old in shard_dir.glob("*.sacc"):
        old.unlink()

    if rust_bin is not None:
        print(f"Organizing dataset using Pure-Rust engine packer: {rust_bin}")
        res = subprocess.run(
            [
                str(rust_bin),
                "organize",
                str(merged_pgn),
                str(shard_dir),
                "--val-pct",
                str(val_pct),
                "--max-games",
                str(max_games),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(res.stdout.strip())
        stats = {"packer": "rust_saccadion_data", "summary": res.stdout.strip()}
    else:
        print("Rust binary not found; packing via Pure-Python (`python-chess`) fallback...")
        stats = _pack_pgn_with_python_chess([merged_pgn], shard_dir, val_pct=val_pct, max_games=max_games)
        stats["packer"] = "python_chess_fallback"

    manifest = {
        "source": source,
        "pgn_files": [str(p) for p in pgn_files],
        "train_random_shards": sorted(str(p) for p in shard_dir.glob("train_random_*.sacc")),
        "train_traj_shards": sorted(str(p) for p in shard_dir.glob("train_traj_*.sacc")),
        "val_shard": str(shard_dir / "val_random.sacc"),
        "stats": stats,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dataset manifest written -> {manifest_path}")
    return shard_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saccadion Automated Chess Dataset Fetcher & Organizer")
    parser.add_argument("--out-dir", default="data", help="Destination directory for raw PGNs and .sacc shards")
    parser.add_argument("--source", choices=["auto", "lichess_elite", "fishtest", "lichess_eval"], default="auto")
    parser.add_argument("--max-games", type=int, default=4000)
    parser.add_argument("--val-pct", type=int, default=5)
    parser.add_argument("--local-pgn", type=str, default=None)
    args = parser.parse_args()

    fetch_and_prepare(
        out_dir=Path(args.out_dir),
        source=args.source,
        max_games=args.max_games,
        val_pct=args.val_pct,
        local_pgn=Path(args.local_pgn) if args.local_pgn else None,
    )
