#!/usr/bin/env python3
"""
hfswr_gui.py - HF Surface Wave Radar performance prediction tool (GUI)

Run:      python hfswr_gui.py
Headless: python hfswr_gui.py --headless out.png [--csv out.csv]
          [--mode range|pdmap|pdcurves] [--freq 10] [--month 1] [--hour 12]
          [--ssn 100] [--sea-state 3] [--vr 50] [--beam 180] [--cit 10]
          [--land-km 0.5] [--no-sea-clutter] [--no-sky] [--no-iono]
          [--atmos tables|parametric]
          [--vmin 0] [--vmax 60] [--vstep 0.25]

Four stacked charts, varying (1) UTC hour, (2) month, (3) sunspot number,
(4) target radial speed, in one of three display modes:
  range    - maximum continuous detection range, one line per frequency
  pdmap    - probability of detection over range (colour) at one frequency
  pdcurves - Pd versus range curves at one frequency, one curve per step

Requires Python 3.9+, numpy, scipy, matplotlib (Tkinter for the GUI).
Physics: see hfswr_model.py.
"""
from __future__ import annotations

import argparse
import csv
import threading
from dataclasses import asdict

import matplotlib
import numpy as np

from hfswr_model import (ATMOS_MODELS, GROUND_TYPES, MAN_MADE, SEA_STATES, HFSWRModel,
                         RadarParams, bragg_frequency_hz, default_site_path,
                         validate_inputs, parameter_record)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LABELLED_F = (5, 10, 15, 20, 25, 30)
MODES = {"range": "Maximum detection range (all frequencies)",
         "pdmap": "Pd over range - map (one frequency)",
         "pdcurves": "Pd versus range - curves (one frequency)"}
KINDS = ("diurnal", "annual", "sunspot", "speed")
INK = "#333333"


# =============================================================================
# Computation
# =============================================================================
def compute_all(model, p, mode, freq, month, hour, ssn, progress=None):
    freqs = p.freqs_mhz if mode == "range" else [freq]
    for i, f in enumerate(freqs):
        if progress:
            progress(f"Ground-wave propagation {f} MHz ({i + 1}/{len(freqs)})")
        model.prepare(p, f)
    if progress:
        progress("Noise, clutter and detection sweeps")
    out = {"mode": mode, "freq": freq}
    for kind in KINDS:
        if mode == "range":
            out[kind] = model.range_sweep(p, kind, month, hour, ssn)
        else:
            out[kind] = model.pd_sweep(p, kind, freq, month, hour, ssn)
    return out


# =============================================================================
# Plotting
# =============================================================================
def _bragg_speed(f_mhz):
    """Radial speed whose Doppler equals the first-order Bragg line (m/s)."""
    return bragg_frequency_hz(f_mhz) * 299792458.0 / (f_mhz * 1e6) / 2.0


def _titles(p, month, hour, ssn):
    return {
        "diurnal": (f"Diurnal variation - {MONTHS[month - 1]}, SSN {ssn:g}", "Time (UTC, h)",
                    (0, 24), list(range(0, 25, 3)), None),
        "annual": (f"Annual variation - {hour:02d}00 UTC, SSN {ssn:g}", "Month",
                   (1, 12), list(range(1, 13)), MONTHS),
        "sunspot": (f"Solar-cycle variation - {MONTHS[month - 1]}, {hour:02d}00 UTC",
                    "Smoothed sunspot number (R12)", (0, 200), list(range(0, 201, 25)), None),
        "speed": (f"Target radial-speed variation - {MONTHS[month - 1]}, {hour:02d}00 UTC, SSN {ssn:g}",
                  "Target radial speed (m/s, positive = closing)",
                  (p.speed_min_ms, p.speed_max_ms), None, None),
    }


def _style(ax):
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_charts(fig, model, p: RadarParams, res, month, hour, ssn):
    fig.clear()
    axes = fig.subplots(4, 1)
    titles = _titles(p, month, hour, ssn)
    mode = res["mode"]
    if mode == "range":
        _draw_range(fig, axes, p, res, titles)
    elif mode == "pdmap":
        _draw_pdmap(fig, axes, p, res, titles)
    else:
        _draw_pdcurves(fig, axes, p, res, titles)

    effects = []
    if p.sea_clutter:
        effects.append(f"sea clutter SS{p.sea_state} ({SEA_STATES[p.sea_state][0].lower()}), "
                       f"wind from {p.wind_from_deg:g}°")
    if p.sky_interference:
        effects.append(f"skywave interference +{p.sky_interf_excess_db:g} dB")
    if p.iono_clutter:
        effects.append("ionospheric self-clutter")
    fig.suptitle(
        f"HFSWR - {model.noise.name}  |  Pt {p.pt_w / 1e3:g} kW, RCS {p.rcs_m2:g} m², "
        f"h {p.target_h_m:g} m, Vr {p.target_vr_ms:g} m/s (charts 1-3), CIT {p.cit_s:g} s, "
        f"beam {p.rx_beamwidth_deg:g}°, Pd {p.pd_req:g} / Pfa {p.pfa:g} "
        f"(SINR {p.snr_req_db:.1f} dB), {p.man_made} noise, "
        f"atmospheric {'P.372 tables' if p.atmos_model.startswith('ITU') else 'parametric'}, "
        f"land {p.land_km:g} km\n"
        + ("Effects: " + "; ".join(effects) if effects else "Effects: noise only"),
        fontsize=8.5, x=0.02, ha="left", color=INK)
    fig.subplots_adjust(left=0.08, right=0.84, top=0.935, bottom=0.05, hspace=0.62)


def _set_x(ax, kind, titles):
    title, xlabel, xlim, xticks, xtl = titles[kind]
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xtl:
        ax.set_xticklabels(xtl)


def _mark_bragg(ax, p, freqs, vertical_lines):
    """Show where targets compete with first-order sea echo on the speed chart."""
    if not p.sea_clutter:
        return
    vb = [_bragg_speed(f) for f in freqs]
    lo, hi = min(vb), max(vb)
    for sgn in (1, -1):
        a, b = sorted((sgn * lo, sgn * hi))
        if b < p.speed_min_ms or a > p.speed_max_ms:
            continue
        if vertical_lines:
            ax.axvline(sgn * vb[0], color="#E8590C", lw=1.0, ls=":")
        else:
            ax.axvspan(max(a, p.speed_min_ms), min(b, p.speed_max_ms), color="#E8590C",
                       alpha=0.12, lw=0)
    txt = (f"Bragg speed ±{vb[0]:.1f} m/s" if vertical_lines
           else f"Bragg speeds {lo:.1f}-{hi:.1f} m/s")
    ax.text(0.99, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color="#B34200")


def _draw_range(fig, axes, p, res, titles):
    freqs = np.array(p.freqs_mhz, dtype=float)
    norm = matplotlib.colors.Normalize(vmin=freqs.min(), vmax=max(freqs.max(), freqs.min() + 1))
    cmap = matplotlib.colormaps["viridis"]
    ymax = max(max(np.max(v) for v in res[k][1].values()) for k in KINDS)
    ymax = max(10.0, 10 * np.ceil(ymax * 1.08 / 10))
    for ax, kind in zip(axes, KINDS):
        x, data = res[kind]
        for f in p.freqs_mhz:
            y = data[f]
            lab = f in LABELLED_F
            ax.plot(x, y, color=cmap(norm(f)), lw=1.6 if lab else 1.0,
                    alpha=1.0 if lab else 0.85, gid=f"{f} MHz")
            if lab:
                ax.annotate(f"{f} MHz", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=7, color=INK)
        _set_x(ax, kind, titles)
        ax.set_ylabel("Detection range (km)", fontsize=9)
        ax.set_ylim(0, ymax)
        if kind == "speed":
            _mark_bragg(ax, p, p.freqs_mhz, vertical_lines=False)
        _style(ax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.06)
    cb.set_label("Frequency (MHz)", fontsize=9)
    cb.ax.tick_params(labelsize=8)


def _pd_ylim(res, p):
    top = 0.0
    for k in KINDS:
        _, r, pd = res[k]
        cols = np.nonzero(np.max(pd, axis=0) > 0.02)[0]
        if len(cols):
            top = max(top, r[cols[-1]])
    return max(50.0, min(p.r_max_km, 10 * np.ceil(top * 1.25 / 10)))


def _draw_pdmap(fig, axes, p, res, titles):
    f = res["freq"]
    ymax = _pd_ylim(res, p)
    cmap = matplotlib.colormaps["Blues"]
    mesh = None
    for ax, kind in zip(axes, KINDS):
        x, r, pd = res[kind]
        mesh = ax.pcolormesh(x, r, pd.T, cmap=cmap, vmin=0, vmax=1, shading="nearest",
                             rasterized=True)
        if np.nanmax(pd) >= p.pd_req and np.nanmin(pd) <= p.pd_req:
            cs = ax.contour(x, r, pd.T, levels=[p.pd_req], colors="#E8590C", linewidths=1.6)
            ax.clabel(cs, fmt={p.pd_req: f"Pd {p.pd_req:g}"}, fontsize=8, colors="#E8590C")
        _set_x(ax, kind, titles)
        ax.set_title(titles[kind][0] + f"  -  {f:g} MHz", fontsize=10, loc="left", color=INK)
        ax.set_ylabel("Range (km)", fontsize=9)
        ax.set_ylim(p.r_min_km, ymax)
        if kind == "speed":
            _mark_bragg(ax, p, [f], vertical_lines=True)
        ax.tick_params(labelsize=8)
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.03)
    cb.set_label("Probability of detection", fontsize=9)
    cb.ax.tick_params(labelsize=8)


def _speed_picks(p, f):
    """Representative speeds: sweep limits, the Bragg speed and round values."""
    cand = {p.speed_min_ms, p.speed_max_ms, round(_bragg_speed(f), 1)}
    cand |= {v for v in (0, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300)}
    vals = sorted(v for v in cand if p.speed_min_ms <= v <= p.speed_max_ms)
    vb = round(_bragg_speed(f), 1)
    if len(vals) > 9:
        idx = np.unique(np.round(np.linspace(0, len(vals) - 1, 9)).astype(int))
        keep = [vals[i] for i in idx]
        if vb in vals and vb not in keep:          # always show the Bragg speed
            keep[int(np.argmin([abs(v - vb) for v in keep[1:-1]])) + 1] = vb
        vals = sorted(keep)
    return vals


def _draw_pdcurves(fig, axes, p, res, titles):
    f = res["freq"]
    xmax = _pd_ylim(res, p)
    picks = {"diurnal": (list(range(0, 24, 3)), "twilight_shifted", lambda v: f"{int(v):02d}00 UTC"),
             "annual": (list(range(1, 13)), "twilight_shifted", lambda v: MONTHS[int(v) - 1]),
             "sunspot": (list(range(0, 201, 25)), "viridis", lambda v: f"SSN {int(v)}"),
             "speed": (_speed_picks(p, f), "viridis", lambda v: f"{v:g} m/s")}
    for ax, kind in zip(axes, KINDS):
        x, r, pd = res[kind]
        vals, cmap_name, fmt = picks[kind]
        cmap = matplotlib.colormaps[cmap_name]
        span = (max(vals) - min(vals)) or 1
        for v in vals:
            i = int(np.argmin(np.abs(x - v)))
            frac = (v - min(vals)) / span * (0.9 if cmap_name != "viridis" else 1.0)
            ax.plot(r, pd[i], color=cmap(frac), lw=1.4, label=fmt(v), gid=fmt(v))
        ax.axhline(p.pd_req, color=INK, lw=0.8, ls="--")
        ax.set_title(titles[kind][0] + f"  -  {f:g} MHz", fontsize=10, loc="left", color=INK)
        ax.set_xlabel("Range (km)", fontsize=9)
        ax.set_ylabel("Probability of detection", fontsize=9)
        ax.set_xlim(p.r_min_km, xmax)
        ax.set_ylim(0, 1.02)
        _style(ax)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)


def write_csv(path, model, p, res, month, hour, ssn):
    """
    CSV with a complete parameter record followed by the chart data.
    Section 1: section, parameter, value, unit, description (every input,
    display setting, constant and derived value). Section 2: one block per chart.
    """
    rows = parameter_record(model, p, res["mode"], res["freq"], month, hour, ssn)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["# HFSWR performance prediction - parameter record"])
        w.writerow(["section", "parameter", "value", "unit", "description"])
        for r in rows:
            w.writerow(r)
        w.writerow([])
        w.writerow(["# Chart data"])
        names = {"diurnal": "utc_hour", "annual": "month", "sunspot": "ssn",
                 "speed": "radial_speed_ms"}
        for kind in KINDS:
            w.writerow([])
            if res["mode"] == "range":
                x, data = res[kind]
                w.writerow([kind, names[kind]] + [f"{f} MHz range (km)" for f in p.freqs_mhz])
                for i, xv in enumerate(x):
                    w.writerow(["", f"{xv:g}"] + [f"{data[f][i]:.1f}" for f in p.freqs_mhz])
            else:
                x, r, pd = res[kind]
                w.writerow([f"{kind} Pd at {res['freq']} MHz", f"{names[kind]} \\ range (km)"]
                           + [f"{v:g}" for v in r])
                for i, xv in enumerate(x):
                    w.writerow(["", f"{xv:g}"] + [f"{v:.4f}" for v in pd[i]])


# =============================================================================
# GUI
# =============================================================================
def run_gui(test_hook=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    model = HFSWRModel(default_site_path())
    root = tk.Tk()
    root.title(f"HFSWR performance prediction - {model.noise.name}")
    root.geometry("1480x980")

    left = ttk.Frame(root, padding=6)
    left.pack(side=tk.LEFT, fill=tk.Y)
    plotf = ttk.Frame(root)
    plotf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    nb = ttk.Notebook(left)
    nb.pack(fill=tk.X)
    tabs = {}
    for name in ("Radar", "Target", "Path & noise", "Sea clutter", "Skywave", "Display"):
        fr = ttk.Frame(nb, padding=6)
        nb.add(fr, text=name)
        tabs[name] = fr

    d = RadarParams()
    V = {}          # tk variables by GUI key
    W = {}          # entry widgets by GUI key (for highlighting)
    LABEL = {}      # field label by GUI key
    TAB = {}        # tab name by GUI key
    style = ttk.Style()
    style.configure("Bad.TEntry", fieldbackground="#FFD6D6", foreground="#9B0000")
    style.configure("Warn.TEntry", fieldbackground="#FFF0C2", foreground="#7A5200")

    def entry(tab, key, label, value):
        fr = ttk.Frame(tabs[tab]); fr.pack(fill=tk.X, pady=1)
        ttk.Label(fr, text=label, width=34).pack(side=tk.LEFT)
        V[key] = tk.StringVar(value=str(value))
        W[key] = ttk.Entry(fr, textvariable=V[key], width=11)
        W[key].pack(side=tk.RIGHT)
        LABEL[key], TAB[key] = label, tab
        V[key].trace_add("write", lambda *_a, k=key: W[k].configure(style="TEntry"))

    def combo(tab, key, label, values, value, width=34):
        fr = ttk.Frame(tabs[tab]); fr.pack(fill=tk.X, pady=2)
        ttk.Label(fr, text=label).pack(anchor="w")
        V[key] = tk.StringVar(value=value)
        ttk.Combobox(fr, textvariable=V[key], values=values, state="readonly",
                     width=width).pack(anchor="w")

    def check(tab, key, label, value):
        V[key] = tk.BooleanVar(value=value)
        ttk.Checkbutton(tabs[tab], text=label, variable=V[key]).pack(anchor="w", pady=2)

    def note(tab, text):
        ttk.Label(tabs[tab], text=text, wraplength=330, foreground="#555").pack(anchor="w", pady=(6, 0))

    # Radar
    entry("Radar", "pt_kw", "Transmit power (kW)", d.pt_w / 1e3)
    entry("Radar", "duty", "Duty cycle (0-1)", d.duty)
    entry("Radar", "gain_dbi", "Tx/Rx gain incl. ground (dBi)", d.gain_dbi)
    entry("Radar", "cit_s", "Coherent integration time (s)", d.cit_s)
    entry("Radar", "bandwidth_khz", "Waveform bandwidth (kHz)", d.bandwidth_hz / 1e3)
    entry("Radar", "beam", "Rx azimuth beamwidth (deg)", d.rx_beamwidth_deg)
    entry("Radar", "look", "Boresight azimuth (deg true)", d.look_az_deg)
    entry("Radar", "losses_db", "System/processing losses (dB)", d.losses_db)
    entry("Radar", "rx_nf_db", "Receiver noise figure (dB)", d.rx_nf_db)
    entry("Radar", "dr_db", "Doppler sidelobe / dyn. range floor (dB)", d.doppler_dr_db)
    entry("Radar", "pd_req", "Required Pd", d.pd_req)
    entry("Radar", "pfa", "Pfa", d.pfa)
    combo("Radar", "swerling", "Target fluctuation model",
          ["Swerling 0 (steady)", "Swerling 1"], "Swerling 0 (steady)")
    note("Radar", "Beamwidth 180° = single receive element (as briefed): clutter is "
                  "collected from the whole seaward half-plane.")
    # Target
    entry("Target", "rcs_m2", "RCS (m²)", d.rcs_m2)
    entry("Target", "target_h_m", "Altitude (m)", d.target_h_m)
    entry("Target", "vr", "Radial speed (m/s)", d.target_vr_ms)
    note("Target", "Radial speed sets the target Doppler 2v/λ relative to the Bragg "
                   "lines (±0.23 Hz at 5 MHz, ±0.56 Hz at 30 MHz). Below ~15 m/s the "
                   "target competes with the second-order sea-echo continuum.")
    # Path & noise
    combo("Path & noise", "sea", "Sea", list(GROUND_TYPES), d.sea)
    combo("Path & noise", "land", "Land between radar and shore", list(GROUND_TYPES), d.land)
    entry("Path & noise", "land_km", "Land section length (km)", d.land_km)
    entry("Path & noise", "k_factor", "Effective earth-radius factor", d.k_factor)
    entry("Path & noise", "r_max_km", "Maximum range computed (km)", d.r_max_km)
    combo("Path & noise", "man_made", "Man-made noise environment", list(MAN_MADE), d.man_made)
    combo("Path & noise", "percentile", "Noise statistic", ["Median", "Upper decile (90 %)"], d.percentile)
    combo("Path & noise", "atmos_model", "Atmospheric noise", list(ATMOS_MODELS), d.atmos_model)
    check("Path & noise", "gal", "Ionospheric screening of galactic noise (f ≤ foF2)", True)
    # Sea clutter
    check("Sea clutter", "sea_clutter", "Include sea clutter (Bragg + second order)", True)
    combo("Sea clutter", "sea_state", "Sea state (WMO)",
          [f"{k} - {v[0]} (Hs {v[1]:g} m)" for k, v in SEA_STATES.items()],
          f"{d.sea_state} - {SEA_STATES[d.sea_state][0]} (Hs {SEA_STATES[d.sea_state][1]:g} m)")
    entry("Sea clutter", "wind", "Wind direction, from (deg true)", d.wind_from_deg)
    entry("Sea clutter", "bragg_sd", "Bragg line broadening s.d. (Hz)", d.bragg_sd_hz)
    note("Sea clutter", "Fully developed Pierson-Moskowitz sea. First-order Bragg "
                        "sigma0 = 2^6 π k0^4 S(2k0) (Barrick); second-order continuum "
                        "scaled by (k0 h)²/2.")
    # Skywave
    check("Skywave", "sky", "Include skywave interference", True)
    entry("Skywave", "sky_db", "Interference above P.372 noise, band open (dB)", d.sky_interf_excess_db)
    check("Skywave", "iono", "Include ionospheric self-clutter", True)
    entry("Skywave", "zen_db", "Antenna gain towards ionosphere (dB rel.)", d.zenith_gain_db)
    entry("Skywave", "refl_db", "F-layer reflection/scatter loss (dB)", d.iono_refl_loss_db)
    entry("Skywave", "iono_sd", "Ionospheric clutter Doppler s.d. (Hz)", d.iono_doppler_sd_hz)
    entry("Skywave", "iono_ext", "Self-clutter range extent beyond hmF2 (km)", d.iono_extent_km)
    note("Skywave", "Interference applies when f < MUF(3000)F2 at the site (P.1239 maps), "
                    "less daytime D-layer absorption. Self-clutter applies when f < foF2, "
                    "at ranges hmF2 to hmF2 + extent. Both are engineering models with "
                    "tunable parameters - calibrate against site data where possible.")
    # Display
    combo("Display", "mode", "Display format", list(MODES.values()), MODES["range"], width=40)
    entry("Display", "freq", "Frequency for Pd displays (MHz)", 10)
    combo("Display", "month", "Month (diurnal & SSN charts)", MONTHS, "Jan", width=8)
    entry("Display", "hour", "UTC hour (annual & SSN charts)", 12)
    entry("Display", "ssn", "Sunspot number (diurnal & annual)", 100)
    entry("Display", "freqs", "Frequencies MHz (start,stop,step)", "5,30,1")
    entry("Display", "vmin", "Speed chart: minimum radial speed (m/s)", d.speed_min_ms)
    entry("Display", "vmax", "Speed chart: maximum radial speed (m/s)", d.speed_max_ms)
    entry("Display", "vstep", "Speed chart: step (m/s)", d.speed_step_ms)
    note("Display", "Charts 1-3 use the target radial speed on the Target tab; "
                    "chart 4 sweeps it. Negative speeds = opening targets.")

    status = tk.StringVar(value="Ready - press Compute")
    btns = ttk.Frame(left); btns.pack(fill=tk.X, pady=(10, 2))
    compute_btn = ttk.Button(btns, text="Compute"); compute_btn.pack(side=tk.LEFT)
    png_btn = ttk.Button(btns, text="Save PNG"); png_btn.pack(side=tk.LEFT, padx=4)
    csv_btn = ttk.Button(btns, text="Save CSV"); csv_btn.pack(side=tk.LEFT)
    ttk.Label(left, textvariable=status, wraplength=340, foreground="#444").pack(anchor="w", pady=6)
    hover = tk.StringVar(value="")
    ttk.Label(left, textvariable=hover, wraplength=340).pack(anchor="w")

    fig = Figure(figsize=(10, 12), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=plotf)
    NavigationToolbar2Tk(canvas, plotf)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    state = {"res": None, "p": None}

    # GUI key <-> RadarParams / validation field names
    FIELD = {"pt_kw": "pt_w", "bandwidth_khz": "bandwidth_hz", "beam": "rx_beamwidth_deg",
             "look": "look_az_deg", "dr_db": "doppler_dr_db", "vr": "target_vr_ms",
             "wind": "wind_from_deg", "bragg_sd": "bragg_sd_hz", "sky_db": "sky_interf_excess_db",
             "zen_db": "zenith_gain_db", "refl_db": "iono_refl_loss_db",
             "iono_sd": "iono_doppler_sd_hz", "iono_ext": "iono_extent_km",
             "vmin": "speed_min_ms", "vmax": "speed_max_ms", "vstep": "speed_step_ms"}
    GUIKEY = {v: k for k, v in FIELD.items()}

    def read_params():
        """Parse every field. Returns (args, parse_errors)."""
        bad = []

        def g(k):
            txt = V[k].get().strip()
            try:
                v = float(txt)
            except ValueError:
                bad.append((k, f"{LABEL[k]}: '{txt}' is not a number"))
                return float("nan")
            if not np.isfinite(v):
                bad.append((k, f"{LABEL[k]}: must be finite"))
            return v

        freqs = []
        try:
            f0, f1, fs = (float(x) for x in V["freqs"].get().split(","))
            if fs <= 0 or f1 < f0:
                raise ValueError
            freqs = [round(v, 3) for v in np.arange(f0, f1 + 1e-9, fs)]
            freqs = [int(v) if float(v).is_integer() else v for v in freqs]
        except ValueError:
            bad.append(("freqs", f"{LABEL['freqs']}: enter start,stop,step with stop ≥ start "
                                 "and step > 0, e.g. 5,30,1"))
        mode = next(k for k, v in MODES.items() if v == V["mode"].get())
        fsel = g("freq")
        if np.isfinite(fsel) and float(fsel).is_integer():
            fsel = int(fsel)
        hour = g("hour")
        p = RadarParams(
            pt_w=g("pt_kw") * 1e3, duty=g("duty"), gain_dbi=g("gain_dbi"), cit_s=g("cit_s"),
            bandwidth_hz=g("bandwidth_khz") * 1e3, rx_beamwidth_deg=g("beam"),
            look_az_deg=g("look"), losses_db=g("losses_db"), rx_nf_db=g("rx_nf_db"),
            doppler_dr_db=g("dr_db"), pd_req=g("pd_req"), pfa=g("pfa"),
            swerling=0 if V["swerling"].get().startswith("Swerling 0") else 1,
            rcs_m2=g("rcs_m2"), target_h_m=g("target_h_m"), target_vr_ms=g("vr"),
            sea=V["sea"].get(), land=V["land"].get(), land_km=g("land_km"),
            k_factor=g("k_factor"), r_max_km=g("r_max_km"), man_made=V["man_made"].get(),
            percentile=V["percentile"].get(), atmos_model=V["atmos_model"].get(),
            galactic_screening=V["gal"].get(),
            sea_clutter=V["sea_clutter"].get(), sea_state=int(V["sea_state"].get()[0]),
            wind_from_deg=g("wind"), bragg_sd_hz=g("bragg_sd"),
            sky_interference=V["sky"].get(), sky_interf_excess_db=g("sky_db"),
            iono_clutter=V["iono"].get(), zenith_gain_db=g("zen_db"),
            iono_refl_loss_db=g("refl_db"), iono_doppler_sd_hz=g("iono_sd"),
            iono_extent_km=g("iono_ext"), speed_min_ms=g("vmin"), speed_max_ms=g("vmax"),
            speed_step_ms=g("vstep"), freqs_mhz=freqs)
        month = MONTHS.index(V["month"].get()) + 1
        return (p, mode, fsel, month, hour, g("ssn")), bad

    def highlight(items, look="Bad.TEntry"):
        for k in W:
            W[k].configure(style="TEntry")
        first = None
        for field, _ in items:
            k = GUIKEY.get(field, field)
            if k in W:
                W[k].configure(style=look)
                first = first or k
        if first:
            nb.select(tabs[TAB[first]])
            W[first].focus_set()

    def on_compute():
        args, bad = read_params()
        if bad:
            highlight(bad)
            messagebox.showerror("Input error", "Please correct:\n\n" + "\n".join(m for _, m in bad))
            return
        p, mode, fsel, month, hour, ssn = args
        errors, warns = validate_inputs(p, month, hour, ssn, fsel if mode != "range" else None,
                                        site=(model.noise.lat, model.noise.lon))
        if errors:
            highlight(errors)
            messagebox.showerror("Input out of range",
                                 "Please correct:\n\n" + "\n".join(f"• {m}" for _, m in errors))
            return
        highlight([])
        if warns:
            highlight(warns, "Warn.TEntry")
            if not messagebox.askokcancel(
                    "Check inputs", "These inputs are allowed but stretch the model:\n\n"
                    + "\n".join(f"• {m}" for _, m in warns) + "\n\nCompute anyway?"):
                return
        args = (p, mode, fsel, month, int(hour), ssn)
        compute_btn.state(["disabled"])

        def work():
            try:
                res = compute_all(model, p, mode, fsel, month, int(hour), ssn,
                                  progress=lambda s: root.after(0, status.set, s))
                root.after(0, finish, args, res)
            except Exception as exc:
                root.after(0, lambda e=exc: (status.set(f"Error: {e}"),
                                             compute_btn.state(["!disabled"])))
        threading.Thread(target=work, daemon=True).start()

    def finish(args, res):
        p, mode, fsel, month, hour, ssn = args
        draw_charts(fig, model, p, res, month, hour, ssn)
        canvas.draw()
        state.update(res=res, p=p, month=month, hour=hour, ssn=ssn)
        extra = ""
        if mode != "range":
            extra = (f"  Bragg lines at ±{bragg_frequency_hz(fsel):.3f} Hz; target Doppler "
                     f"{2 * p.target_vr_ms * fsel * 1e6 / 299792458.0:.2f} Hz.")
        status.set(f"Done. Required SINR {p.snr_req_db:.1f} dB." + extra +
                   " Ground-wave results are cached.")
        compute_btn.state(["!disabled"])

    def on_move(ev):
        if ev.inaxes is None or state["res"] is None or ev.xdata is None:
            return
        res = state["res"]
        if res["mode"] == "pdmap":
            kind = KINDS[[a for a in fig.axes[:4]].index(ev.inaxes)] if ev.inaxes in fig.axes[:4] else None
            if kind:
                x, r, pd = res[kind]
                i = int(np.argmin(np.abs(x - ev.xdata))); j = int(np.argmin(np.abs(r - ev.ydata)))
                hover.set(f"x = {x[i]:g}, range = {r[j]:.0f} km, Pd = {pd[i, j]:.2f}")
            return
        best = None
        for ln in ev.inaxes.get_lines():
            if ln.get_gid() is None:
                continue
            xd, yd = (np.asarray(a) for a in ln.get_data())
            i = int(np.argmin(np.abs(xd - ev.xdata)))
            dist = abs(yd[i] - ev.ydata)
            if best is None or dist < best[0]:
                best = (dist, ln.get_gid(), xd[i], yd[i])
        if best:
            ylab = "range" if res["mode"] == "range" else "Pd"
            hover.set(f"{best[1]}:  x = {best[2]:g},  {ylab} = {best[3]:.2f}")

    def on_png():
        if state["res"] is None: return
        fn = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if fn: fig.savefig(fn, dpi=150)

    def on_csv():
        if state["res"] is None: return
        fn = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if fn:
            write_csv(fn, model, state["p"], state["res"], state["month"], state["hour"], state["ssn"])
            status.set(f"Saved {fn} (full parameter record + chart data)")

    compute_btn.configure(command=on_compute)
    png_btn.configure(command=on_png)
    csv_btn.configure(command=on_csv)
    canvas.mpl_connect("motion_notify_event", on_move)
    if test_hook:      # used by the automated GUI smoke test only
        root.after(300, test_hook, {"root": root, "V": V, "W": W, "compute": on_compute,
                                    "status": status, "fig": fig, "state": state,
                                    "save_csv": on_csv})
    root.mainloop()


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", metavar="PNG")
    ap.add_argument("--csv")
    ap.add_argument("--mode", choices=list(MODES), default="range")
    ap.add_argument("--freq", type=float, default=10)
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--hour", type=int, default=12)
    ap.add_argument("--ssn", type=float, default=100)
    ap.add_argument("--sea-state", type=int, default=None)
    ap.add_argument("--vr", type=float, default=None)
    ap.add_argument("--beam", type=float, default=None)
    ap.add_argument("--cit", type=float, default=None)
    ap.add_argument("--land-km", type=float, default=None)
    ap.add_argument("--no-sea-clutter", action="store_true")
    ap.add_argument("--no-sky", action="store_true")
    ap.add_argument("--no-iono", action="store_true")
    ap.add_argument("--atmos", choices=["tables", "parametric"], default="tables",
                    help="Atmospheric noise: ITU-R P.372 tables or parametric approximation")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--vstep", type=float, default=None)
    a = ap.parse_args()
    if not a.headless:
        run_gui()
        return
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    p = RadarParams()
    for attr, val in (("sea_state", a.sea_state), ("target_vr_ms", a.vr),
                      ("rx_beamwidth_deg", a.beam), ("cit_s", a.cit), ("land_km", a.land_km),
                      ("speed_min_ms", a.vmin), ("speed_max_ms", a.vmax), ("speed_step_ms", a.vstep)):
        if val is not None:
            setattr(p, attr, val)
    p.sea_clutter = not a.no_sea_clutter
    p.sky_interference = not a.no_sky
    p.iono_clutter = not a.no_iono
    p.atmos_model = ATMOS_MODELS[0] if a.atmos == "tables" else ATMOS_MODELS[1]
    freq = int(a.freq) if float(a.freq).is_integer() else a.freq
    model = HFSWRModel(default_site_path())
    errors, warns = validate_inputs(p, a.month, a.hour, a.ssn, freq if a.mode != "range" else None,
                                    site=(model.noise.lat, model.noise.lon))
    for _, m in warns:
        print(f"WARNING: {m}")
    if errors:
        for _, m in errors:
            print(f"ERROR: {m}")
        raise SystemExit(2)
    res = compute_all(model, p, a.mode, freq, a.month, a.hour, a.ssn)
    fig = Figure(figsize=(11, 16), dpi=110)
    draw_charts(fig, model, p, res, a.month, a.hour, a.ssn)
    fig.savefig(a.headless, dpi=110)
    if a.csv:
        write_csv(a.csv, model, p, res, a.month, a.hour, a.ssn)
    print(f"Saved {a.headless}")


if __name__ == "__main__":
    main()
