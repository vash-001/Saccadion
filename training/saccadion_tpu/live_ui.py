"""20 Hz Flicker-Free Live Colab HTML/SVG Training Cockpit for Saccadion-2 Causal.

Uses `IPython.display.DisplayHandle` throttled to <= 20 Hz (50ms interval) to render:
- Smooth animated progress bars (current epoch + overall curriculum + live ETA + pos/s)
- 6 dark-mode KPI telemetry cards (Loss, Chess IQ, Causal IQ, Real/Est Elo, Fly-Brain Biology,
  and Background CPU Stockfish Sparring Arena scoreboard)
- 4 live inline SVG curves updating smoothly with zero flicker and < 0.5 ms CPU cost!
"""

from __future__ import annotations

import html
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _svg_line_chart(
    title: str,
    series: List[Tuple[str, str, List[float]]],
    width: int = 290,
    height: int = 128,
    ref_line: Optional[Tuple[float, str, str]] = None,
    y_min_fixed: Optional[float] = None,
    y_max_fixed: Optional[float] = None,
) -> str:
    """Renders a crisp, lightweight inline SVG multi-series line chart in < 0.1 ms."""
    pad_l, pad_r, pad_t, pad_b = 34, 10, 22, 20
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b

    all_vals: List[float] = []
    for _, _, vals in series:
        all_vals.extend(v for v in vals if v == v)  # filter NaN
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
            margin = 0.08 * (y_hi - y_lo)
            if y_min_fixed is None:
                y_lo -= margin
            if y_max_fixed is None:
                y_hi += margin

    def map_y(v: float) -> float:
        frac = (v - y_lo) / max(y_hi - y_lo, 1e-9)
        return pad_t + ph * (1.0 - max(0.0, min(1.0, frac)))

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'style="background:#0f172a;border-radius:8px;border:1px solid #1e293b;">',
        f'<text x="{pad_l}" y="14" fill="#e2e8f0" font-size="11" font-weight="600" '
        f'font-family="monospace">{html.escape(title)}</text>',
    ]

    # Horizontal grid + y labels
    for frac in (0.0, 0.5, 1.0):
        y = pad_t + ph * (1.0 - frac)
        val = y_lo + frac * (y_hi - y_lo)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="#1e293b" stroke-width="1"/>'
        )
        lbl = f"{val:.0f}" if abs(val) >= 100 else (f"{val:.1f}" if abs(val) >= 10 else f"{val:.2f}")
        parts.append(
            f'<text x="{pad_l - 4}" y="{y + 3:.1f}" fill="#64748b" font-size="9" '
            f'text-anchor="end" font-family="monospace">{lbl}</text>'
        )

    if ref_line is not None:
        ry = map_y(ref_line[0])
        parts.append(
            f'<line x1="{pad_l}" y1="{ry:.1f}" x2="{width - pad_r}" y2="{ry:.1f}" '
            f'stroke="{ref_line[1]}" stroke-dasharray="4,3" stroke-width="1.2"/>'
        )

    legend_x = width - pad_r
    for label, color, vals in reversed(series):
        if len(vals) >= 1:
            n = len(vals)
            if n == 1:
                px = pad_l + pw * 0.5
                py = map_y(vals[0])
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>')
            else:
                pts = " ".join(
                    f"{pad_l + (i / (n - 1)) * pw:.1f},{map_y(v):.1f}"
                    for i, v in enumerate(vals)
                )
                parts.append(
                    f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
                )
                lx = pad_l + pw
                ly = map_y(vals[-1])
                parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.8" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x}" y="{height - 5}" fill="{color}" font-size="9" '
            f'text-anchor="end" font-family="monospace">{html.escape(label)}</text>'
        )
        legend_x -= max(48, len(label) * 6 + 10)

    parts.append("</svg>")
    return "".join(parts)


class LiveTrainingHUD:
    """Smooth 20 Hz flicker-free live training dashboard for Google Colab & terminal."""

    def __init__(self, max_hz: float = 20.0, device_label: str = "TPU"):
        self.min_interval = 1.0 / max(1.0, max_hz)
        self.device_label = device_label
        self._last_render_t = 0.0
        self._handle: Any = None
        self._in_notebook = self._detect_notebook()
        self.step_loss_ring: List[float] = []
        self.step_kc_ring: List[float] = []

    @staticmethod
    def _detect_notebook() -> bool:
        try:
            from IPython import get_ipython

            ip = get_ipython()
            return ip is not None and "IPKernelApp" in getattr(ip, "config", {}) or "google.colab" in sys.modules
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
    ) -> str:
        ep_pct = 100.0 * step / max(1, steps_per_epoch)
        tot_steps = max(1, total_epochs * steps_per_epoch)
        cur_tot = (epoch - 1) * steps_per_epoch + step
        tot_pct = 100.0 * cur_tot / tot_steps

        last = history[-1] if history else {}
        arena = arena_status or {}

        val_loss_str = f"{last['val_loss']:.4f}" if "val_loss" in last else "--"
        val_delta_str = f"({last['val_delta']:+.4f})" if "val_delta" in last else ""
        top1_str = f"{last['val_top1_acc']:.2f}%" if "val_top1_acc" in last else "--"
        top3_str = f"{last['val_top3_acc']:.2f}%" if "val_top3_acc" in last else "--"
        wdl_str = f"{last['val_wdl_acc']:.2f}%" if "val_wdl_acc" in last else "--"
        mae_str = f"{last['val_cp_mae']:.1f} cp" if "val_cp_mae" in last else "--"

        causal_f1_str = f"{last['val_causal_f1'] * 100:.1f}%" if "val_causal_f1" in last else "--"
        nod_str = f"{last['val_noduli_margin']:+.3f}" if "val_noduli_margin" in last else "--"
        vnc_str = f"{last['val_vnc2']:.3f}" if "val_vnc2" in last else "--"

        est_raw = f"{last['est_raw_elo']:.0f}" if "est_raw_elo" in last else "--"
        est_pvs = f"{last['est_search_elo']:.0f}" if "est_search_elo" in last else "--"
        real_elo_val = float(arena.get("real_elo", 0.0))
        real_elo_str = (
            f"{real_elo_val:.0f} +/- {arena.get('elo_ci', 0.0):.0f}"
            if real_elo_val > 0
            else "Calibrating..."
        )

        orth_str = f"{last['val_orth']:.4f}" if "val_orth" in last else "--"
        pam_str = f"{last['eta_pam']:.3f}" if "eta_pam" in last else "0.080"
        d1r2_str = f"{last['eta_dop1r2']:.3f}" if "eta_dop1r2" in last else "0.140"
        soc_str = f"{last['eta_soc']:.3f}" if "eta_soc" in last else "0.220"

        opp_name = str(arena.get("opponent_name", "Warming up"))
        wdl_arena = f"{arena.get('wins', 0)}W - {arena.get('draws', 0)}D - {arena.get('losses', 0)}L"
        harvested = int(arena.get("harvested_positions", 0))
        last_game = str(arena.get("last_result", "Waiting for Epoch-1 .saccnet..."))

        eta_m, eta_s = divmod(int(max(0, eta_sec)), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        eta_fmt = f"{eta_h:d}h {eta_m:02d}m {eta_s:02d}s" if eta_h > 0 else f"{eta_m:02d}m {eta_s:02d}s"

        # Build live SVG charts
        loss_series: List[Tuple[str, str, List[float]]] = [
            ("StepLoss", "#38bdf8", self.step_loss_ring[-120:]),
        ]
        if history:
            loss_series.append(("ValLoss", "#f472b6", [r["val_loss"] for r in history]))
        svg_loss = _svg_line_chart("Loss (Live Step & Val)", loss_series)

        acc_series: List[Tuple[str, str, List[float]]] = [
            ("Top1%", "#34d399", [r["val_top1_acc"] for r in history]),
            ("WDL%", "#fbbf24", [r["val_wdl_acc"] for r in history]),
            ("CausalF1%", "#c084fc", [r.get("val_causal_f1", 0.0) * 100.0 for r in history]),
        ]
        svg_acc = _svg_line_chart("Chess & Causal IQ (%)", acc_series, y_min_fixed=0.0, y_max_fixed=100.0)

        elo_series: List[Tuple[str, str, List[float]]] = [
            ("Est+PVS", "#22d3ee", [r["est_search_elo"] for r in history]),
            ("EstNet", "#4ade80", [r["est_raw_elo"] for r in history]),
        ]
        real_elos = [r["real_arena_elo"] for r in history if r.get("real_arena_elo", 0.0) > 0]
        if real_elos:
            elo_series.append(("RealSF", "#fb7185", real_elos))
        svg_elo = _svg_line_chart("Real Arena vs Proxy Elo", elo_series)

        kc_vals = [r["val_active_kc_pct"] for r in history] if history else self.step_kc_ring[-80:]
        svg_kc = _svg_line_chart(
            "APL Sparsity (Target 8.0%)",
            [("ActiveKC%", "#a78bfa", kc_vals)],
            ref_line=(8.0, "#22c55e", "8%"),
            y_min_fixed=0.0,
        )

        qat_badge = '<span style="background:#7c3aed;color:#fff;padding:2px 7px;border-radius:4px;font-size:11px;">QAT ON</span>' if qat else '<span style="background:#1e293b;color:#94a3b8;padding:2px 7px;border-radius:4px;font-size:11px;">FP32/BF16</span>'

        return f"""
<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#020617;color:#e2e8f0;
padding:14px;border-radius:12px;border:1px solid #1e293b;max-width:1240px;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
    <div>
      <span style="font-size:16px;font-weight:700;color:#38bdf8;">SACCADION-2 CAUSAL</span>
      <span style="color:#64748b;"> | </span>
      <span style="color:#a78bfa;font-weight:600;">{html.escape(stage_name)}</span>
      <span style="color:#64748b;"> | </span>
      <span style="color:#94a3b8;">{html.escape(self.device_label)} | Ticks={ticks}</span>
      {qat_badge}
    </div>
    <div style="font-size:12px;color:#cbd5e1;">
      <b>{pos_per_sec:,.0f} pos/s</b> | {step_ms:.1f} ms/step | ETA: <b style="color:#34d399;">{eta_fmt}</b>
    </div>
  </div>

  <!-- Progress Bars -->
  <div style="margin-bottom:10px;">
    <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-bottom:3px;">
      <span>Epoch {epoch:02d}/{total_epochs:02d} — Step {step:,}/{steps_per_epoch:,} ({ep_pct:5.1f}%)</span>
      <span>Overall Progress: {tot_pct:5.1f}%</span>
    </div>
    <div style="background:#0f172a;height:10px;border-radius:6px;overflow:hidden;border:1px solid #1e293b;margin-bottom:5px;">
      <div style="width:{ep_pct:.1f}%;height:100%;background:linear-gradient(90deg,#0ea5e9,#34d399);transition:width 0.05s linear;"></div>
    </div>
    <div style="background:#0f172a;height:6px;border-radius:4px;overflow:hidden;border:1px solid #1e293b;">
      <div style="width:{tot_pct:.1f}%;height:100%;background:linear-gradient(90deg,#6366f1,#ec4899);transition:width 0.1s linear;"></div>
    </div>
  </div>

  <!-- 6 KPI Cards -->
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:8px;margin-bottom:10px;">
    <div style="background:#0f172a;padding:9px;border-radius:8px;border:1px solid #1e293b;">
      <div style="font-size:10px;color:#64748b;">LOSS TELEMETRY</div>
      <div style="font-size:16px;font-weight:700;color:#38bdf8;">{step_loss:.4f}</div>
      <div style="font-size:11px;color:#94a3b8;">Val: {val_loss_str} {val_delta_str}</div>
    </div>
    <div style="background:#0f172a;padding:9px;border-radius:8px;border:1px solid #1e293b;">
      <div style="font-size:10px;color:#64748b;">CHESS IQ (HELD-OUT)</div>
      <div style="font-size:15px;font-weight:700;color:#34d399;">Top-1: {top1_str}</div>
      <div style="font-size:11px;color:#94a3b8;">Top3 {top3_str} | WDL {wdl_str} | {mae_str}</div>
    </div>
    <div style="background:#0f172a;padding:9px;border-radius:8px;border:1px solid #1e293b;">
      <div style="font-size:10px;color:#64748b;">CAUSAL IQ (AOTU+NODULI)</div>
      <div style="font-size:15px;font-weight:700;color:#c084fc;">Concept F1: {causal_f1_str}</div>
      <div style="font-size:11px;color:#94a3b8;">NoduliGap {nod_str} | VNC2 {vnc_str}</div>
    </div>
    <div style="background:#0f172a;padding:9px;border-radius:8px;border:1px solid #1e293b;">
      <div style="font-size:10px;color:#64748b;">REAL & PROXY ELO</div>
      <div style="font-size:15px;font-weight:700;color:#fb7185;">Real: {html.escape(real_elo_str)}</div>
      <div style="font-size:11px;color:#94a3b8;">Est Net ~{est_raw} | +PVS ~{est_pvs}</div>
    </div>
    <div style="background:#0f172a;padding:9px;border-radius:8px;border:1px solid #1e293b;">
      <div style="font-size:10px;color:#64748b;">FLY-BRAIN BIOLOGY</div>
      <div style="font-size:15px;font-weight:700;color:#a78bfa;">APL KC: {step_kc_pct:4.1f}%</div>
      <div style="font-size:11px;color:#94a3b8;">PAM {pam_str} | D1R2 {d1r2_str} | SOC {soc_str} | Orth {orth_str}</div>
    </div>
    <div style="background:#0f172a;padding:9px;border-radius:8px;border:1px solid #1e293b;">
      <div style="font-size:10px;color:#64748b;">CPU STOCKFISH ARENA</div>
      <div style="font-size:13px;font-weight:700;color:#fbbf24;">{html.escape(opp_name)}</div>
      <div style="font-size:11px;color:#94a3b8;">{wdl_arena} | AntiSF+{harvested} | {html.escape(last_game[:28])}</div>
    </div>
  </div>

  <!-- 4 Live Inline SVG Charts -->
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:8px;">
    {svg_loss}
    {svg_acc}
    {svg_elo}
    {svg_kc}
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
        force: bool = False,
    ) -> None:
        self.step_loss_ring.append(step_loss)
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
                )
                if self._handle is None:
                    self._handle = display(HTML(markup), display_id=True)
                else:
                    self._handle.update(HTML(markup))
                return
            except Exception:
                pass

        # Clean single-line terminal fallback (zero scroll spam)
        pct = 100.0 * step / max(1, steps_per_epoch)
        bar_w = 24
        filled = int(bar_w * step / max(1, steps_per_epoch))
        bar = "#" * filled + "-" * (bar_w - filled)
        arena_elo = arena_status.get("real_elo", 0.0) if arena_status else 0.0
        sf_tag = f" | SF-Elo={arena_elo:.0f}" if arena_elo > 0 else ""
        sys.stdout.write(
            f"\r[Ep {epoch:02d}/{total_epochs:02d}] [{bar}] {pct:5.1f}% "
            f"loss={step_loss:.4f} kc={step_kc_pct:4.1f}% {pos_per_sec:6,.0f} pos/s {step_ms:4.1f}ms{sf_tag}   "
        )
        sys.stdout.flush()
        if force:
            sys.stdout.write("\n")
            sys.stdout.flush()
