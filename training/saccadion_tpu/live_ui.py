"""Zorynto v4.0 (Hakuji x Akane) Live Training Cockpit for Saccadion-7 Neuro-Apex.

Architectural design system derived directly from Zorynto v4.0 tokens:
  - 6-color palette: Paper (#FBF9F7), Bone (#F1ECE8), Stone (#B8AEA9),
    Umber (#4A403D), Ink (#1D1918), and rare Akane (#8F3A35).
  - Zero drop shadows; layers separated strictly by 1px hairline rules & tonal steps.
  - Nested radii (26 -> 18 -> 10 -> 6 -> pill) & tabular numerals.
  - Throttled cleanly to 2 Hz (500 ms) so Google Colab's WebSocket NEVER stalls!

Displays:
  1. Top Architectural Bar with S. Akane mark, Freeze/Thaw phase pill, throughput & dual progress
  2. 6 Zorynto KPI Bento Cards (Loss/EMA, Chess IQ, Causal/Blunder/Brilliant, Real Elo Inverse Hero,
     Freeze/Thaw Neurodevelopment Matrix, Fly-Brain Biology)
  3. 6 Inline SVG Telemetry Charts (Loss, Real vs Proxy Elo, GF/Deep MAE, Blunder/Brilliant/Causal %,
     Chess Accuracy %, APL GABA Sparsity)
  4. Full-Width Live Multi-Engine Arena Boardroom with 3 live 8x8 SVG chessboards, per-engine W/D/L
     hairline table, Zorynto 60-game result strip, and recent games feed
  5. All 12 Drosophila Connectome Circuits live status grid with dotted leaders
"""

from __future__ import annotations

import html
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Exact Zorynto v4.0 Hakuji x Akane palette
Z_PAPER = "#FBF9F7"
Z_BONE = "#F1ECE8"
Z_STONE = "#B8AEA9"
Z_UMBER = "#4A403D"
Z_INK = "#1D1918"
Z_AKANE = "#8F3A35"
Z_LINE_SUB = "rgba(29,25,24,0.13)"
Z_LINE = "rgba(29,25,24,0.24)"
Z_AKANE_WASH = "rgba(143,58,53,0.08)"

_FEN_PIECES = {
    "K": "&#9812;", "Q": "&#9813;", "R": "&#9814;", "B": "&#9815;", "N": "&#9816;", "P": "&#9817;",
    "k": "&#9818;", "q": "&#9819;", "r": "&#9820;", "b": "&#9821;", "n": "&#9822;", "p": "&#9823;",
}


def _svg_mini_chessboard(fen: str, flip: bool = False, size: int = 144) -> str:
    """Renders a crisp Zorynto-palette 8x8 SVG chessboard directly from a FEN string in < 0.08 ms."""
    board_part = fen.split()[0] if fen else "8/8/8/8/8/8/8/8"
    rows = board_part.split("/")
    grid: List[List[str]] = []
    for r_str in rows[:8]:
        row_cells: List[str] = []
        for ch in r_str:
            if ch.isdigit():
                row_cells.extend([""] * int(ch))
            else:
                row_cells.append(ch)
        while len(row_cells) < 8:
            row_cells.append("")
        grid.append(row_cells[:8])
    while len(grid) < 8:
        grid.append([""] * 8)

    sq = size / 8.0
    parts = [
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
        f'style="display:block;border-radius:8px;border:1px solid {Z_LINE};flex:none;">'
    ]
    for vr in range(8):
        for vf in range(8):
            br = (7 - vr) if flip else vr
            bf = (7 - vf) if flip else vf
            is_dark = ((br + bf) & 1) == 1
            fill = "#DDD4CE" if is_dark else Z_PAPER
            x = vf * sq
            y = vr * sq
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{sq:.1f}" height="{sq:.1f}" fill="{fill}"/>')
            pc = grid[br][bf]
            if pc and pc in _FEN_PIECES:
                p_col = Z_INK if pc.islower() else Z_UMBER
                parts.append(
                    f'<text x="{x + sq * 0.5:.1f}" y="{y + sq * 0.78:.1f}" text-anchor="middle" '
                    f'font-size="{sq * 0.80:.1f}" fill="{p_col}" font-family="serif">{_FEN_PIECES[pc]}</text>'
                )
    parts.append("</svg>")
    return "".join(parts)


def _svg_line_chart(
    title: str,
    series: List[Tuple[str, str, List[float]]],
    width: int = 340,
    height: int = 148,
    ref_line: Optional[Tuple[float, str, str]] = None,
    y_min_fixed: Optional[float] = None,
    y_max_fixed: Optional[float] = None,
    subtitle: str = "",
) -> str:
    """Renders a crisp Zorynto Hakuji x Akane inline SVG multi-series line chart in < 0.1 ms."""
    pad_l, pad_r, pad_t, pad_b = 38, 12, 26, 22
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b

    all_vals: List[float] = []
    for _, _, vals in series:
        all_vals.extend(v for v in vals if v == v)
    if ref_line is not None:
        all_vals.append(ref_line[0])

    if not all_vals:
        y_lo, y_hi = 0.0, 1.0
    else:
        y_lo = min(all_vals) if y_min_fixed is None else y_min_fixed
        y_hi = max(all_vals) if y_max_fixed is None else y_max_fixed
        if abs(y_hi - y_lo) < 1e-6:
            y_lo -= 0.5
            y_hi += 0.5
        else:
            margin = 0.10 * (y_hi - y_lo)
            if y_min_fixed is None:
                y_lo -= margin
            if y_max_fixed is None:
                y_hi += margin

    def map_y(v: float) -> float:
        frac = (v - y_lo) / max(y_hi - y_lo, 1e-9)
        return pad_t + ph * (1.0 - max(0.0, min(1.0, frac)))

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'style="display:block;background:{Z_PAPER};border-radius:10px;border:1px solid {Z_LINE_SUB};">',
        f'<text x="{pad_l}" y="15" fill="{Z_INK}" font-size="10.5" font-weight="800" '
        f'letter-spacing="0.08em" font-family="ui-monospace,Consolas,monospace">{html.escape(title.upper())}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{width - pad_r}" y="15" fill="{Z_UMBER}" font-size="9.5" text-anchor="end" '
            f'font-family="ui-monospace,Consolas,monospace">{html.escape(subtitle)}</text>'
        )

    for frac in (0.0, 0.5, 1.0):
        y = pad_t + ph * (1.0 - frac)
        val = y_lo + frac * (y_hi - y_lo)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="{Z_LINE_SUB}" stroke-width="1"/>'
        )
        lbl = f"{val:.0f}" if abs(val) >= 100 else (f"{val:.1f}" if abs(val) >= 10 else f"{val:.2f}")
        parts.append(
            f'<text x="{pad_l - 5}" y="{y + 3:.1f}" fill="{Z_UMBER}" font-size="9" '
            f'text-anchor="end" font-family="ui-monospace,Consolas,monospace">{lbl}</text>'
        )

    if ref_line is not None:
        ry = map_y(ref_line[0])
        parts.append(
            f'<line x1="{pad_l}" y1="{ry:.1f}" x2="{width - pad_r}" y2="{ry:.1f}" '
            f'stroke="{ref_line[1]}" stroke-dasharray="4,3" stroke-width="1.2"/>'
        )

    legend_x = width - pad_r
    for idx_s, (label, color, vals) in enumerate(reversed(series)):
        if len(vals) >= 1:
            n = len(vals)
            if n == 1:
                px = pad_l + pw * 0.5
                py = map_y(vals[0])
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{color}"/>')
            else:
                pts = " ".join(
                    f"{pad_l + (i / (n - 1)) * pw:.1f},{map_y(v):.1f}"
                    for i, v in enumerate(vals)
                )
                dash = ' stroke-dasharray="4,2"' if idx_s >= 2 else ""
                parts.append(
                    f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} '
                    f'stroke-linecap="round" stroke-linejoin="round" points="{pts}"/>'
                )
                lx = pad_l + pw
                ly = map_y(vals[-1])
                parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.8" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x}" y="{height - 6}" fill="{color}" font-size="9" font-weight="700" '
            f'text-anchor="end" font-family="ui-monospace,Consolas,monospace">{html.escape(label)}</text>'
        )
        legend_x -= max(50, len(label) * 5.8 + 10)

    parts.append("</svg>")
    return "".join(parts)


class LiveTrainingHUD:
    """Zero-lag Zorynto v4.0 live training cockpit for Google Colab & terminal."""

    def __init__(self, max_hz: float = 2.0, device_label: str = "TPU"):
        # Hard-cap at <= 4 Hz so Google Colab Jupyter IOPub never buffers or freezes!
        self.min_interval = 1.0 / max(0.5, min(max_hz, 4.0))
        self.device_label = device_label
        self._last_render_t = 0.0
        self._handle: Any = None
        self._in_notebook = self._detect_notebook()
        self.step_loss_ring: List[float] = []
        self.step_kc_ring: List[float] = []
        self._ema_loss: Optional[float] = None

    @staticmethod
    def _detect_notebook() -> bool:
        try:
            from IPython import get_ipython

            ip = get_ipython()
            return (ip is not None and "IPKernelApp" in getattr(ip, "config", {})) or "google.colab" in sys.modules
        except Exception:
            return False

    def render_html(
        self,
        epoch: int,
        total_epochs: int,
        step: int,
        steps_per_epoch: int,
        stage_name: str,
        ticks: int,
        qat: bool,
        step_loss: float,
        step_kc_pct: float,
        step_ms: float,
        pos_per_sec: float,
        eta_sec: float,
        history: List[Dict[str, Any]],
        arena_status: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        ep_pct = 100.0 * step / max(1, steps_per_epoch)
        tot_steps = max(1, total_epochs * steps_per_epoch)
        cur_tot = (epoch - 1) * steps_per_epoch + step
        tot_pct = 100.0 * cur_tot / tot_steps

        last = history[-1] if history else {}
        arena = arena_status or {}
        ex = extra or {}

        val_loss_str = f"{last['val_loss']:.4f}" if "val_loss" in last else "--"
        val_delta_str = f"({last['val_delta']:+.4f})" if "val_delta" in last else ""
        best_val = ex.get("best_val_loss", last.get("best_val_loss"))
        best_val_str = f"{best_val:.4f}" if isinstance(best_val, (int, float)) and best_val < 1e8 else "--"
        ema_badge = "EMA WIN" if ex.get("used_ema", last.get("used_ema", False)) else "LIVE"

        top1_str = f"{last['val_top1_acc']:.2f}%" if "val_top1_acc" in last else "--"
        top3_str = f"{last['val_top3_acc']:.2f}%" if "val_top3_acc" in last else "--"
        wdl_str = f"{last['val_wdl_acc']:.2f}%" if "val_wdl_acc" in last else "--"
        mae_str = f"{last['val_cp_mae']:.1f} cp" if "val_cp_mae" in last else "--"
        gf_mae_v = ex.get("gf_quiet_mae_live", last.get("val_gf_quiet_mae"))
        gf_mae_str = f"{gf_mae_v:.1f} cp" if isinstance(gf_mae_v, (int, float)) and gf_mae_v > 0 else "--"

        causal_f1_str = f"{last['val_causal_f1'] * 100:.1f}%" if "val_causal_f1" in last else "--"
        nod_str = f"{last['val_noduli_margin']:+.3f}" if "val_noduli_margin" in last else "--"
        imb_str = f"{last['val_imbalance']:.4f}" if "val_imbalance" in last else "--"

        bl_avoid = ex.get("blunder_avoid_pct", last.get("blunder_avoid_pct", 0.0))
        wm_bl = ex.get("wm_blunder_pct", last.get("wm_blunder_pct", 0.0))
        br_t1 = ex.get("bril_top1_pct", last.get("bril_top1_pct", 0.0))
        br_t3 = ex.get("bril_top3_pct", last.get("bril_top3_pct", 0.0))
        bl_seen = int(ex.get("blunder_seen", 0))
        br_seen = int(ex.get("bril_seen", 0))
        retro_rec = float(ex.get("retro_goal_recall", last.get("val_retro_recall", 0.0) * 100.0))

        est_raw = f"{last['est_raw_elo']:.0f}" if "est_raw_elo" in last else "--"
        est_pvs = f"{last['est_search_elo']:.0f}" if "est_search_elo" in last else "--"
        real_elo_val = float(arena.get("real_elo", 0.0))
        real_elo_str = (
            f"{real_elo_val:.0f} +/- {arena.get('elo_ci', 0.0):.0f}"
            if real_elo_val > 0
            else "Calibrating..."
        )
        elo_level = "L5 APEX" if real_elo_val >= 2500 else ("L4 MASTER" if real_elo_val >= 2200 else "L3 LIVE")

        orth_str = f"{last['val_orth']:.4f}" if "val_orth" in last else "--"
        pam_str = f"{last['eta_pam']:.3f}" if "eta_pam" in last else "0.080"
        d1r2_str = f"{last['eta_dop1r2']:.3f}" if "eta_dop1r2" in last else "0.140"
        soc_str = f"{last['eta_soc']:.3f}" if "eta_soc" in last else "0.220"

        eta_m, eta_s = divmod(int(max(0, eta_sec)), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        eta_fmt = f"{eta_h:d}h {eta_m:02d}m {eta_s:02d}s" if eta_h > 0 else f"{eta_m:02d}m {eta_s:02d}s"

        # 6 Inline SVG Telemetry Charts (Zorynto Hakuji x Akane)
        loss_series: List[Tuple[str, str, List[float]]] = [("StepEMA", Z_INK, self.step_loss_ring[-120:])]
        val_losses = [r["val_loss"] for r in history if "val_loss" in r]
        gf_losses = [r["val_gf_loss"] for r in history if "val_gf_loss" in r]
        if val_losses:
            loss_series.append(("Val", Z_AKANE, val_losses))
        if gf_losses:
            loss_series.append(("GF", Z_STONE, gf_losses))
        svg_loss = _svg_line_chart("Loss Telemetry", loss_series, subtitle=f"best {best_val_str}")

        elo_series: List[Tuple[str, str, List[float]]] = []
        pvs_elos = [r["est_search_elo"] for r in history if "est_search_elo" in r]
        raw_elos = [r["est_raw_elo"] for r in history if "est_raw_elo" in r]
        real_elos = [r["real_arena_elo"] for r in history if r.get("real_arena_elo", 0.0) > 0]
        if real_elos:
            elo_series.append(("RealArena", Z_AKANE, real_elos))
        elif real_elo_val > 0:
            elo_series.append(("RealArena", Z_AKANE, [real_elo_val]))
        if pvs_elos:
            elo_series.append(("Est+PVS", Z_INK, pvs_elos))
        if raw_elos:
            elo_series.append(("EstNet", Z_STONE, raw_elos))
        svg_elo = _svg_line_chart("Real Arena vs Proxy Elo", elo_series, subtitle=f"now {real_elo_str}")

        mae_series: List[Tuple[str, str, List[float]]] = []
        gf_maes = [r["val_gf_quiet_mae"] for r in history if r.get("val_gf_quiet_mae", 0.0) > 0]
        dp_maes = [r["val_cp_mae"] for r in history if r.get("val_cp_mae", 0.0) > 0]
        if gf_maes:
            mae_series.append(("GF Quiet", Z_AKANE, gf_maes))
        elif isinstance(gf_mae_v, (int, float)) and gf_mae_v > 0:
            mae_series.append(("GF Quiet", Z_AKANE, [float(gf_mae_v)]))
        if dp_maes:
            mae_series.append(("Deep B", Z_INK, dp_maes))
        svg_mae = _svg_line_chart("Eval Precision (MAE cp)", mae_series, ref_line=(50.0, Z_AKANE, "50cp"), subtitle=f"GF {gf_mae_str}")

        causal_series: List[Tuple[str, str, List[float]]] = []
        bl_vals = [r["blunder_avoid_pct"] for r in history if "blunder_avoid_pct" in r]
        wm_vals = [r["wm_blunder_pct"] for r in history if "wm_blunder_pct" in r]
        br_vals = [r["bril_top3_pct"] for r in history if "bril_top3_pct" in r]
        f1_vals = [r["val_causal_f1"] * 100.0 for r in history if "val_causal_f1" in r]
        causal_series.append(("Blunder%", Z_INK, bl_vals if bl_vals else ([bl_avoid] if bl_seen else [])))
        causal_series.append(("WM-Drop%", Z_AKANE, wm_vals if wm_vals else ([wm_bl] if bl_seen else [])))
        causal_series.append(("Bril!!T3%", Z_UMBER, br_vals if br_vals else ([br_t3] if br_seen else [])))
        causal_series.append(("16-F1%", Z_STONE, f1_vals))
        svg_causal = _svg_line_chart("Causal, Blunder & Brilliant!! (%)", causal_series, y_min_fixed=0.0, y_max_fixed=100.0)

        acc_series: List[Tuple[str, str, List[float]]] = [
            ("WDL%", Z_INK, [r.get("val_wdl_acc", 0.0) for r in history]),
            ("Top3%", Z_UMBER, [r.get("val_top3_acc", 0.0) for r in history]),
            ("Top1%", Z_AKANE, [r.get("val_top1_acc", 0.0) for r in history]),
        ]
        svg_acc = _svg_line_chart("Chess Accuracy IQ (%)", acc_series, y_min_fixed=0.0, y_max_fixed=100.0)

        kc_vals = [r.get("val_active_kc_pct", 0.0) for r in history] if history else self.step_kc_ring[-80:]
        svg_kc = _svg_line_chart(
            "APL GABA Sparsity (Target 8%)",
            [("ActiveKC%", Z_INK, kc_vals)],
            ref_line=(8.0, Z_AKANE, "8%"),
            y_min_fixed=0.0,
            subtitle=f"{step_kc_pct:.1f}%",
        )

        # Freeze / Thaw Module Pills
        freeze_map: Dict[str, str] = ex.get("freeze_map") or {
            "GF+HalfKA": "THAWED 1.0x",
            "Optic+GNN": "THAWED 1.0x",
            "CX+World": "WARM 0.6x",
            "Meta-DA": "FROZEN 0.0x",
        }
        ft_pills = []
        for mod_name, state in freeze_map.items():
            if "THAWED" in state:
                p_bg, p_fg, p_bd = Z_INK, Z_PAPER, Z_INK
            elif "FROZEN" in state:
                p_bg, p_fg, p_bd = "transparent", Z_UMBER, Z_LINE
            elif "ANCHOR" in state:
                p_bg, p_fg, p_bd = Z_PAPER, Z_INK, Z_LINE
            else:
                p_bg, p_fg, p_bd = Z_AKANE_WASH, Z_AKANE, Z_AKANE
            border_style = "dashed" if "FROZEN" in state else "solid"
            ft_pills.append(
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 9px;'
                f'border-radius:999px;border:1px {border_style} {p_bd};background:{p_bg};color:{p_fg};font-size:10.5px;font-weight:700;">'
                f'<span>{html.escape(mod_name)}</span><span style="opacity:0.9;">{html.escape(state)}</span></div>'
            )
        freeze_html = "".join(ft_pills)

        # Full-Width Live Multi-Engine Arena Boardroom
        active_boards: Dict[str, Dict[str, Any]] = arena.get("active_boards") or {}
        board_cards = []
        for wid_key in sorted(active_boards.keys())[:3]:
            b = active_boards[wid_key]
            sacc_w = bool(b.get("sacc_white", True))
            side_lbl = "WHITE" if sacc_w else "BLACK"
            cp_v = int(b.get("last_cp", 0))
            cp_str = f"{cp_v / 100.0:+.2f}"
            moves_tail = " ".join(b.get("recent_moves", [])[-6:]) or "opening..."
            svg_b = _svg_mini_chessboard(str(b.get("fen", "")), flip=not sacc_w, size=136)
            board_cards.append(
                f'<div style="display:flex;gap:12px;align-items:center;padding:10px;border-radius:12px;'
                f'border:1px solid {Z_LINE_SUB};background:{Z_PAPER};min-width:0;">'
                f'{svg_b}'
                f'<div style="min-width:0;flex:1;display:grid;gap:4px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">'
                f'<span style="font-size:10px;font-weight:800;letter-spacing:0.12em;color:{Z_AKANE};">BOARD {html.escape(wid_key)}</span>'
                f'<span style="font-size:10px;padding:1px 7px;border-radius:999px;background:{Z_INK};color:{Z_PAPER};font-weight:700;">SACC {side_lbl}</span>'
                f'</div>'
                f'<div style="font-size:13px;font-weight:800;color:{Z_INK};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                f'vs {html.escape(str(b.get("opponent", "Engine")))}</div>'
                f'<div style="font-size:11px;color:{Z_UMBER};">Elo <b>{int(b.get("opp_elo", 0))}</b> | Ply <b>{int(b.get("ply", 0))}</b> | Eval <b style="color:{Z_AKANE if cp_v < -50 else Z_INK};">{cp_str}</b></div>'
                f'<div style="font-size:10px;color:{Z_UMBER};background:{Z_BONE};padding:4px 7px;border-radius:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                f'{html.escape(moves_tail)}</div>'
                f'</div></div>'
            )
        if not board_cards:
            board_cards.append(
                f'<div style="padding:14px;border-radius:12px;border:1px dashed {Z_LINE};background:{Z_PAPER};'
                f'font-size:12px;color:{Z_UMBER};text-align:center;">'
                f'Live 3-Board Multi-Engine Arena warming up — boards go live automatically right after Epoch-1 .saccnet export!</div>'
            )

        # Per-Engine Family Hairline Scoreboard Table
        per_eng: Dict[str, List[int]] = arena.get("per_engine_wdl") or {}
        eng_rows = []
        for fam, wdl in sorted(per_eng.items()):
            w_c, d_c, l_c = int(wdl[0]), int(wdl[1]), int(wdl[2])
            tot_g = max(1, w_c + d_c + l_c)
            pct = 100.0 * (w_c + 0.5 * d_c) / tot_g
            eng_rows.append(
                f'<tr><td style="padding:6px 8px;border-top:1px solid {Z_LINE_SUB};font-weight:800;color:{Z_INK};">{html.escape(fam)}</td>'
                f'<td style="padding:6px 8px;border-top:1px solid {Z_LINE_SUB};color:{Z_UMBER};">{w_c + d_c + l_c}</td>'
                f'<td style="padding:6px 8px;border-top:1px solid {Z_LINE_SUB};color:{Z_INK};font-weight:700;">{w_c}W - {d_c}D - {l_c}L</td>'
                f'<td style="padding:6px 8px;border-top:1px solid {Z_LINE_SUB};font-weight:800;color:{Z_AKANE if pct < 45 else Z_INK};">{pct:5.1f}%</td></tr>'
            )
        if not eng_rows:
            eng_rows.append(
                f'<tr><td colspan="4" style="padding:8px;color:{Z_UMBER};font-size:11px;">Waiting for first completed match...</td></tr>'
            )

        # Zorynto 60-Game Result Strip (.sbars: Ink = Win, Stone = Draw, Akane shorter = Loss)
        strip: List[int] = arena.get("result_strip") or []
        strip_cells = []
        for pad_i in range(max(0, 60 - len(strip))):
            strip_cells.append(
                f'<span style="display:block;height:100%;border-radius:3px;border:1px dashed {Z_LINE_SUB};"></span>'
            )
        for r_val in strip[-60:]:
            if r_val > 0:
                strip_cells.append(f'<span style="display:block;height:100%;border-radius:3px;background:{Z_INK};"></span>')
            elif r_val == 0:
                strip_cells.append(f'<span style="display:block;height:82%;border-radius:3px;background:{Z_STONE};"></span>')
            else:
                strip_cells.append(f'<span style="display:block;height:66%;border-radius:3px;background:{Z_AKANE};"></span>')
        strip_html = "".join(strip_cells)

        # Recent Completed Matches Feed
        recent_games: List[Dict[str, Any]] = arena.get("recent_games") or []
        feed_items = []
        for g in reversed(recent_games[-6:]):
            res = str(g.get("res", "DRAW"))
            r_bg = Z_INK if res == "WIN" else (Z_AKANE if res == "LOSS" else Z_BONE)
            r_fg = Z_PAPER if res in ("WIN", "LOSS") else Z_INK
            feed_items.append(
                f'<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;'
                f'padding:4px 0;border-top:1px solid {Z_LINE_SUB};font-size:11px;">'
                f'<span><span style="display:inline-block;padding:1px 6px;border-radius:999px;background:{r_bg};color:{r_fg};font-size:9.5px;font-weight:800;margin-right:6px;">{res}</span>'
                f'<b>G{int(g.get("gnum", 0))}</b> vs {html.escape(str(g.get("opponent", "")))} ({int(g.get("opp_elo", 0))})</span>'
                f'<span style="color:{Z_UMBER};">{int(g.get("plies", 0))}p | W{int(g.get("worker_id", 0))}</span></div>'
            )
        if not feed_items:
            feed_items.append(f'<div style="font-size:11px;color:{Z_UMBER};padding:4px 0;">{html.escape(str(arena.get("last_result", "Waiting for Epoch-1 export...")))}</div>')

        # All 12 Drosophila Brain Circuits with Zorynto dotted leaders
        def _cbadge(name: str, val_str: str, alive: bool) -> str:
            dot_col = Z_AKANE if alive else Z_STONE
            return (
                f'<div style="display:flex;align-items:baseline;gap:6px;padding:5px 9px;border-radius:8px;'
                f'border:1px solid {Z_LINE_SUB};background:{Z_PAPER};font-size:10.5px;">'
                f'<span style="width:7px;height:7px;border-radius:50%;background:{dot_col};display:inline-block;flex:none;"></span>'
                f'<span style="color:{Z_INK};font-weight:700;white-space:nowrap;">{html.escape(name)}</span>'
                f'<span style="flex:1;border-bottom:1px dotted {Z_LINE};margin-bottom:2px;min-width:8px;"></span>'
                f'<span style="color:{Z_UMBER};font-weight:700;white-space:nowrap;">{html.escape(val_str)}</span></div>'
            )

        c_shunt = float(last.get("c_shunt_gain", 0.0))
        c_lob = float(last.get("c_lob_focus", 0.0))
        c_mbon = float(last.get("c_mbon_ei", 0.0))
        c_lh = float(last.get("c_lh_veto", 0.0))
        c_rc = float(last.get("c_rc_norm", 0.0))
        c_cx = float(last.get("c_cx_norm", 0.0))
        c_cent = float(last.get("c_cent_focus", 0.0))
        c_plan_h = float(last.get("c_plan_entropy", 0.0))
        c_pfl2 = float(last.get("c_pfl2", 0.0))
        c_jepa = float(last.get("jepa_loss", 0.0))
        c_da = float(last.get("meta_dopamine_gain", 0.0))
        c_f1 = float(last.get("val_causal_f1", 0.0)) * 100.0
        circuits_html = "".join([
            _cbadge("01. Dendritic GF+HalfKA", f"MAE {gf_mae_str}", gf_mae_str != "--"),
            _cbadge("02. FlyHash + APL 92%", f"{step_kc_pct:4.1f}% KC", 1.0 <= step_kc_pct <= 25.0),
            _cbadge("03. Push-Pull MBON 2:1", f"E/I {c_mbon:.2f}", c_mbon > 0.01),
            _cbadge("04. Lateral Horn 98%-I", f"veto {c_lh:.2f}", c_lh > 0.001),
            _cbadge("05. T4/T5 Shunting", f"gain {c_shunt:.2f}", c_shunt > 0.01),
            _cbadge("06. 3L Optic + 6ch RetroGNN", f"R@8 {retro_rec:.0f}% f {c_lob:.2f}", c_lob > 0.02 or retro_rec > 1.0),
            _cbadge("07. Rich-Club 16I+16B", f"||h|| {c_rc:.2f}", c_rc > 0.1),
            _cbadge("08. CX Phasor 16-Sec", f"||cx|| {c_cx:.2f}", c_cx > 0.05),
            _cbadge("09. Centrifugal 72%-I", f"std {c_cent:.2f}", c_cent > 0.005),
            _cbadge("10. K=4 Creative Plans", f"H {c_plan_h:.2f}", c_plan_h > 0.1),
            _cbadge("11. PFL3+Noduli+World", f"WM {wm_bl:.0f}% F1 {c_f1:.0f}%", c_pfl2 > 0.01 or bl_seen > 0),
            _cbadge("12. Dop1R1/2+DPM+SOC", f"DA {c_da:+.3f} J {c_jepa:.2f}", c_jepa > 0 or c_da != 0),
        ])

        qat_pill = (
            f'<span style="padding:2px 9px;border-radius:999px;background:{Z_AKANE};color:{Z_PAPER};font-size:10.5px;font-weight:800;">STE-QAT ON</span>'
            if qat
            else f'<span style="padding:2px 9px;border-radius:999px;border:1px solid {Z_LINE};background:{Z_PAPER};color:{Z_UMBER};font-size:10.5px;font-weight:700;">INT8-CACHE / BF16</span>'
        )
        wdl_tot = f"{arena.get('wins', 0)}W - {arena.get('draws', 0)}D - {arena.get('losses', 0)}L"

        return f"""
<div style="font-family:ui-monospace,'IBM Plex Mono',Consolas,monospace;font-variant-numeric:tabular-nums;
background:{Z_PAPER};color:{Z_INK};padding:16px;border-radius:26px;border:1px solid {Z_LINE};max-width:1320px;">

  <!-- ZORYNTO TOP ARCHITECTURAL HEADER -->
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;
  padding-bottom:12px;border-bottom:1px solid {Z_LINE_SUB};margin-bottom:12px;">
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <div style="position:relative;width:34px;height:34px;border-radius:10px;background:{Z_INK};color:{Z_PAPER};
      display:flex;align-items:center;justify-content:center;font-weight:900;font-size:16px;">
        S<span style="position:absolute;right:-3px;top:-3px;width:10px;height:10px;border-radius:50%;
        background:{Z_AKANE};border:2px solid {Z_PAPER};"></span>
      </div>
      <div>
        <div style="font-size:15px;font-weight:900;letter-spacing:-0.02em;color:{Z_INK};">
          SACCADION-8 RETROGRADE FISCHER-GRAPH
          <span style="margin-left:6px;padding:2px 9px;border-radius:999px;background:{Z_INK};color:{Z_PAPER};font-size:10.5px;font-weight:800;">{html.escape(stage_name)}</span>
        </div>
        <div style="font-size:10.5px;color:{Z_UMBER};letter-spacing:0.06em;">
          FLYWIRE DROSOPHILA CONNECTOME | {html.escape(self.device_label)} | TICKS={ticks} | {qat_pill}
        </div>
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:13px;font-weight:800;color:{Z_INK};">
        {pos_per_sec:,.0f} pos/s <span style="color:{Z_STONE};">|</span> {step_ms:.1f} ms/step <span style="color:{Z_STONE};">|</span>
        ETA <span style="color:{Z_AKANE};">{eta_fmt}</span>
      </div>
      <div style="font-size:10.5px;color:{Z_UMBER};">
        Epoch {epoch:02d}/{total_epochs:02d} — Step {step:,}/{steps_per_epoch:,} ({ep_pct:5.1f}%) | Overall {tot_pct:5.1f}%
      </div>
    </div>
  </div>

  <!-- DUAL HAIRLINE PROGRESS BARS (Ink Epoch + Akane Overall) -->
  <div style="display:grid;gap:5px;margin-bottom:14px;">
    <div style="background:{Z_BONE};height:8px;border-radius:999px;overflow:hidden;border:1px solid {Z_LINE_SUB};">
      <div style="width:{ep_pct:.1f}%;height:100%;background:{Z_INK};border-radius:999px;"></div>
    </div>
    <div style="background:{Z_BONE};height:5px;border-radius:999px;overflow:hidden;">
      <div style="width:{tot_pct:.1f}%;height:100%;background:{Z_AKANE};border-radius:999px;"></div>
    </div>
  </div>

  <!-- 6 ZORYNTO KPI BENTO CARDS -->
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:10px;margin-bottom:14px;">
    <!-- 1. Loss & EMA Guard -->
    <div style="background:{Z_BONE};padding:12px 14px;border-radius:16px;border:1px solid {Z_LINE_SUB};">
      <div style="display:flex;justify-content:space-between;font-size:10px;letter-spacing:0.12em;color:{Z_UMBER};font-weight:700;">
        <span>LOSS &amp; EMA GUARD</span><span style="color:{Z_AKANE};">{ema_badge}</span>
      </div>
      <div style="font-size:22px;font-weight:900;letter-spacing:-0.03em;color:{Z_INK};margin:2px 0;">{step_loss:.4f}</div>
      <div style="font-size:11px;color:{Z_UMBER};">Val {val_loss_str} {val_delta_str} | Best <b>{best_val_str}</b></div>
    </div>

    <!-- 2. Chess IQ & GF Quiet Precision -->
    <div style="background:{Z_BONE};padding:12px 14px;border-radius:16px;border:1px solid {Z_LINE_SUB};">
      <div style="font-size:10px;letter-spacing:0.12em;color:{Z_UMBER};font-weight:700;">CHESS IQ &amp; GF PRECISION</div>
      <div style="font-size:21px;font-weight:900;letter-spacing:-0.03em;color:{Z_INK};margin:2px 0;">GF {gf_mae_str}</div>
      <div style="font-size:11px;color:{Z_UMBER};">Top1 <b>{top1_str}</b> | Top3 {top3_str} | WDL {wdl_str} | B {mae_str}</div>
    </div>

    <!-- 3. Causal, Blunder & Brilliant (!!) Mastery -->
    <div style="background:{Z_BONE};padding:12px 14px;border-radius:16px;border:1px solid {Z_LINE_SUB};">
      <div style="display:flex;justify-content:space-between;font-size:10px;letter-spacing:0.12em;color:{Z_UMBER};font-weight:700;">
        <span>CAUSAL / RETRO / !!</span><span>R@8 {retro_rec:4.1f}% | F1 {causal_f1_str}</span>
      </div>
      <div style="font-size:20px;font-weight:900;letter-spacing:-0.03em;color:{Z_AKANE};margin:2px 0;">Avoid {bl_avoid:4.1f}%</div>
      <div style="font-size:11px;color:{Z_UMBER};">WM-Drop <b>{wm_bl:4.1f}%</b> | !! T1 {br_t1:4.1f}% / T3 <b>{br_t3:4.1f}%</b></div>
    </div>

    <!-- 4. REAL MULTI-ENGINE ELO (ZORYNTO INK INVERSE HERO CARD) -->
    <div style="background:{Z_INK};color:{Z_PAPER};padding:12px 14px;border-radius:16px;border:1px solid {Z_INK};">
      <div style="display:flex;justify-content:space-between;align-items:center;font-size:10px;letter-spacing:0.12em;color:{Z_STONE};font-weight:700;">
        <span>REAL ARENA ELO</span>
        <span style="padding:1px 7px;border-radius:999px;background:{Z_AKANE};color:{Z_PAPER};font-size:9.5px;font-weight:800;">{elo_level}</span>
      </div>
      <div style="font-size:21px;font-weight:900;letter-spacing:-0.03em;color:{Z_PAPER};margin:2px 0;">{html.escape(real_elo_str)}</div>
      <div style="font-size:11px;color:{Z_STONE};">{wdl_tot} | Net ~{est_raw} | +PVS ~{est_pvs}</div>
    </div>

    <!-- 5. Freeze / Thaw Neurodevelopment Matrix -->
    <div style="background:{Z_BONE};padding:10px 12px;border-radius:16px;border:1px solid {Z_LINE_SUB};">
      <div style="font-size:10px;letter-spacing:0.12em;color:{Z_UMBER};font-weight:700;margin-bottom:5px;">FREEZE / THAW CURRICULUM</div>
      <div style="display:grid;gap:3px;">{freeze_html}</div>
    </div>

    <!-- 6. Fly-Brain Biology & Piece Imbalance -->
    <div style="background:{Z_BONE};padding:12px 14px;border-radius:16px;border:1px solid {Z_LINE_SUB};">
      <div style="font-size:10px;letter-spacing:0.12em;color:{Z_UMBER};font-weight:700;">FLY-BRAIN &amp; IMBALANCE</div>
      <div style="font-size:21px;font-weight:900;letter-spacing:-0.03em;color:{Z_INK};margin:2px 0;">APL {step_kc_pct:4.1f}%</div>
      <div style="font-size:11px;color:{Z_UMBER};">Imb {imb_str} | Nod {nod_str} | PAM {pam_str} | Orth {orth_str}</div>
    </div>
  </div>

  <!-- 6 INLINE SVG TELEMETRY CHARTS (3x2 ZORYNTO WINDOW) -->
  <div style="background:{Z_BONE};padding:10px;border-radius:20px;border:1px solid {Z_LINE_SUB};margin-bottom:14px;">
    <div style="display:flex;justify-content:space-between;align-items:center;padding:0 4px 8px 4px;font-size:10.5px;letter-spacing:0.12em;color:{Z_UMBER};font-weight:800;">
      <span>TELEMETRY CURVES — REAL-TIME CONVERGENCE &amp; CAUSAL MASTERY</span>
      <span>BLUNDERS {bl_seen:,} | BRILLIANTS!! {br_seen:,}</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:8px;">
      {svg_loss}
      {svg_elo}
      {svg_mae}
      {svg_causal}
      {svg_acc}
      {svg_kc}
    </div>
  </div>

  <!-- FULL-WIDTH LIVE MULTI-ENGINE ARENA BOARDROOM -->
  <div style="background:{Z_BONE};padding:12px;border-radius:20px;border:1px solid {Z_LINE_SUB};margin-bottom:14px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;padding:0 4px 10px 4px;">
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:8px;height:8px;border-radius:50%;background:{Z_AKANE};display:inline-block;"></span>
        <span style="font-size:11px;letter-spacing:0.12em;font-weight:900;color:{Z_INK};">LIVE MULTI-ENGINE ARENA BOARDROOM</span>
        <span style="font-size:10.5px;color:{Z_UMBER};">({html.escape(str(arena.get("engine_mode", "Waiting...")))})</span>
      </div>
      <div style="font-size:11px;color:{Z_INK};font-weight:700;">
        Total: {int(arena.get("games_played", 0))} games | {wdl_tot}
      </div>
    </div>

    <!-- 3 Live Parallel SVG Chessboards -->
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:10px;margin-bottom:10px;">
      {"".join(board_cards)}
    </div>

    <!-- Per-Engine Table + Zorynto 60-Game Result Strip & Recent Matches Feed -->
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px;">
      <div style="background:{Z_PAPER};border-radius:12px;border:1px solid {Z_LINE_SUB};padding:10px 12px;">
        <div style="font-size:10px;letter-spacing:0.12em;color:{Z_UMBER};font-weight:800;margin-bottom:6px;">PER-ENGINE GAUNTLET SCOREBOARD</div>
        <table style="width:100%;border-collapse:collapse;font-size:11.5px;">
          <thead>
            <tr style="text-align:left;font-size:10px;letter-spacing:0.08em;color:{Z_UMBER};">
              <th style="padding:4px 8px;">ENGINE</th><th style="padding:4px 8px;">GAMES</th><th style="padding:4px 8px;">W - D - L</th><th style="padding:4px 8px;">SCORE</th>
            </tr>
          </thead>
          <tbody>{"".join(eng_rows)}</tbody>
        </table>
      </div>

      <div style="background:{Z_PAPER};border-radius:12px;border:1px solid {Z_LINE_SUB};padding:10px 12px;display:grid;gap:8px;align-content:start;">
        <div>
          <div style="display:flex;justify-content:space-between;font-size:10px;letter-spacing:0.12em;color:{Z_UMBER};font-weight:800;margin-bottom:5px;">
            <span>LAST 60 ARENA GAMES (INK=WIN | STONE=DRAW | AKANE=LOSS)</span>
            <span>{len(strip)}/60</span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(60,minmax(0,1fr));gap:2px;height:24px;align-items:end;">
            {strip_html}
          </div>
        </div>
        <div>
          <div style="font-size:10px;letter-spacing:0.12em;color:{Z_UMBER};font-weight:800;margin-bottom:3px;">RECENT COMPLETED MATCHES</div>
          {"".join(feed_items)}
        </div>
      </div>
    </div>
  </div>

  <!-- ALL 12 DROSOPHILA CONNECTOME CIRCUITS STATUS GRID -->
  <div style="background:{Z_BONE};padding:10px 12px;border-radius:16px;border:1px solid {Z_LINE_SUB};">
    <div style="font-size:10px;letter-spacing:0.14em;color:{Z_UMBER};font-weight:800;margin-bottom:6px;">
      ALL 12 DROSOPHILA CONNECTOME CIRCUITS — LIVE TELEMETRY (AKANE DOT = ACTIVE &amp; LEARNING)
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:6px;">
      {circuits_html}
    </div>
  </div>
</div>
"""

    def update(
        self,
        epoch: int,
        total_epochs: int,
        step: int,
        steps_per_epoch: int,
        stage_name: str,
        ticks: int,
        qat: bool,
        step_loss: float,
        step_kc_pct: float,
        step_ms: float,
        pos_per_sec: float,
        eta_sec: float,
        history: List[Dict[str, Any]],
        arena_status: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        self._ema_loss = step_loss if self._ema_loss is None else (0.92 * self._ema_loss + 0.08 * step_loss)
        self.step_loss_ring.append(self._ema_loss)
        self.step_kc_ring.append(step_kc_pct)
        if len(self.step_loss_ring) > 300:
            self.step_loss_ring = self.step_loss_ring[-200:]
            self.step_kc_ring = self.step_kc_ring[-200:]

        now = time.time()
        if not force and (now - self._last_render_t) < self.min_interval:
            return
        self._last_render_t = now

        if self._in_notebook:
            try:
                from IPython.display import HTML, display

                markup = self.render_html(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    step=step,
                    steps_per_epoch=steps_per_epoch,
                    stage_name=stage_name,
                    ticks=ticks,
                    qat=qat,
                    step_loss=step_loss,
                    step_kc_pct=step_kc_pct,
                    step_ms=step_ms,
                    pos_per_sec=pos_per_sec,
                    eta_sec=eta_sec,
                    history=history,
                    arena_status=arena_status,
                    extra=extra,
                )
                if self._handle is None:
                    self._handle = display(HTML(markup), display_id=True)
                else:
                    self._handle.update(HTML(markup))
                return
            except Exception:
                pass

        pct = 100.0 * step / max(1, steps_per_epoch)
        bar_w = 24
        filled = int(bar_w * step / max(1, steps_per_epoch))
        bar = "#" * filled + "-" * (bar_w - filled)
        arena_elo = arena_status.get("real_elo", 0.0) if arena_status else 0.0
        sf_tag = f" | Arena={arena_elo:.0f}" if arena_elo > 0 else ""
        sys.stdout.write(
            f"\r[Ep {epoch:02d}/{total_epochs:02d}] [{bar}] {pct:5.1f}% "
            f"loss={step_loss:.4f} kc={step_kc_pct:4.1f}% {pos_per_sec:6,.0f} pos/s {step_ms:4.1f}ms{sf_tag}   "
        )
        sys.stdout.flush()
        if force:
            sys.stdout.write("\n")
            sys.stdout.flush()
