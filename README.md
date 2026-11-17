# HFSWR performance prediction tool — Land's End (v3.2)

Predicts monostatic HF surface-wave radar performance at 5–30 MHz and plots
four stacked charts — diurnal, annual, solar-cycle and target radial-speed
variability — in one of three display formats:

| Mode | Content |
|---|---|
| `range` | Maximum continuous detection range (first range where Pd < Pd_req), one line per frequency |
| `pdmap` | Pd over range (colour) against hour / month / SSN / speed, one frequency, Pd_req contour |
| `pdcurves` | Pd versus range curves, one frequency, one curve per hour / month / SSN / speed step |

Charts 1–3 use the target radial speed on the Target tab; chart 4 sweeps it
(default 0–60 m/s in 0.25 m/s steps; negative = opening). The speed chart
shades (range mode) or marks (Pd modes) the first-order Bragg speeds. A range
of 0 means the target is not detected even at the minimum range (5 km).

## Files
| File | Purpose |
|---|---|
| `hfswr_gui.py` | Tkinter GUI (tabbed parameters) and `--headless` PNG/CSV mode |
| `hfswr_model.py` | Physics: propagation, noise, sea clutter, skywave, detection |
| `site_lands_end.json` | P.372 noise coefficients, P.1239 foF2 and M(3000)F2 for Land's End |
| `build_site_data.py` | Regenerates the site file for another location from ITU-R SG3 data |

## Run
```
pip install numpy scipy matplotlib
python hfswr_gui.py
python hfswr_gui.py --headless out.png --mode pdmap --freq 10 --sea-state 4 --vr 50 --beam 180 \
       --vmin -30 --vmax 60 --vstep 0.25
```
First `range` run ~60–75 s (ground-wave modes for 26 frequencies, target and
sea-surface heights); cached thereafter. Pd modes need only one frequency (~5 s).

## Model summary
**Detection.** SINR = E_target / (N0 + C_sea(fd) + C_iono(fd)) after coherent
integration; Pd by Marcum (Swerling 0) or Swerling 1 at the chosen Pfa. Default
Pd 0.9, Pfa 1e-6, Swerling 0 ⇒ 13.2 dB.

**Propagation (ITU-R P.368).** Residue series + Norton flat earth, Airy height
gain, Millington mixed path. Validated against GRWAVE to ≤ 0.3 dB over sea.

**Noise (ITU-R P.372).** Port of the ITU P372 reference code (< 0.001 dB);
galactic noise screened when f ≤ foF2. The **Atmospheric noise** control
(Path & noise tab; `--atmos tables|parametric`) selects:

| Option | Method | Accuracy vs P.372 tables (total median noise, Land's End) |
|---|---|---|
| ITU-R P.372 tables (default) | Numerical coefficient maps from the site file | Reference |
| Parametric (approx.) | Closed form: Fam(1 MHz) as a 21-term Fourier series in local time and month; FaA = α(f)·Fam + β(f) with cubic α, β in log f; Du, Dl as smooth functions | Rural 0.4 dB rms; Quiet rural 1.3 dB rms (2.8 dB p95); detection range typically within ±2 % (Rural) or ±5 % (Quiet rural) |

The parametric coefficients are a weighted least-squares fit to the P.372
tables for Land's End over 3–30 MHz. It is smooth through the day (no 4-hour
block steps) but is site-specific: a warning appears if used outside 3–30 MHz
or at a site more than 5° latitude / 10° longitude away. Refit for another
site with `fit_parametric_atmos(SiteNoise(...))` in `hfswr_model.py`.

**Sea clutter (sea state 0–7).**
* Fully developed Pierson–Moskowitz sea; Hs from the WMO sea-state code
  (mid-band values), U19.5 = sqrt(Hs·g/0.21); cos⁴(θ/2) directional spreading;
  selectable wind direction.
* First-order Bragg lines at ±√(g·k0/π) Hz with σ⁰ = 2⁶πk0⁴S(2k0) (Barrick
  1972), averaged over the receive beam. Reproduces the classic saturated
  value of about −17 dB upwind; falls away when Bragg waves are undeveloped
  (low sea state, low frequency).
* Second-order continuum: total energy (k0h)²/2 relative to first order
  (Barrick's wave-height relation with unit weighting, capped at 0.5),
  distributed around each line as (1 + (Δf/fp)²)^-5/2 (fp = dominant wave
  frequency), so the far tail falls as Δf⁻⁵ like the PM spectrum's ω⁻⁵ tail.
* Clutter cell R × beamwidth × c/2B; a Doppler sidelobe / dynamic-range floor
  (default −85 dB) leaks total clutter into every Doppler bin.

**Skywave (engineering models — calibrate against site data).**
* External interference: P.372 median noise + excess (default 10 dB) when
  f < MUF(3000)F2 at the site, less 3× vertical D-layer absorption.
* Ionospheric self-clutter: own signal at near-vertical incidence when
  f < foF2, ranges hmF2 (Shimazaki, from M(3000)F2) to hmF2 + extent,
  Doppler spread about 0 Hz, scaled by antenna gain towards the ionosphere.

## Saved CSV files
Every CSV (GUI **Save CSV** or `--csv`) is self-describing:

1. **Parameter record** — columns `section, parameter, value, unit, description`:
   * *Run*: tool version, UTC timestamp, site name and coordinates
   * *Display*: mode, Pd frequency, fixed month / UTC hour / SSN
   * *Radar, Detection, Target, Path, Noise, Sea clutter, Skywave, Speed
     sweep, Frequencies*: every input parameter
   * *Constants*: ground constants, P.372 man-made and galactic coefficients
   * *Derived*: required SINR, range cell, Doppler bin, Hs and wind speed,
     foF2, M(3000)F2, MUF, hmF2, solar zenith angle
   * *Per frequency*: wavelength, target Doppler, Bragg frequency and speed,
     first/second-order σ⁰, k0h, atmospheric / man-made / galactic / total Fa,
     skywave weightings, D-layer absorption (at the fixed month/hour/SSN)
2. **Chart data** — one block per chart (range per frequency, or the Pd
   matrix over range).

`PARAM_META` in `hfswr_model.py` documents every input; the writer refuses to
run if a new parameter is added without an entry, so nothing can be silently
omitted. The file is UTF-8 with BOM so Excel shows units such as m² correctly.

## Input range checking
Every input is checked before computing (GUI and headless):
* **Parse errors** — non-numeric or malformed fields (e.g. the frequency list).
* **Hard limits** — values outside physical or model validity are rejected
  (table below); the offending fields turn red and the GUI jumps to their tab.
* **Consistency checks** — e.g. max range > min range + 10 steps, land section
  shorter than the minimum range, speed-sweep max > min, ≤ 60 frequencies,
  ≤ 2000 speed points.
* **Warnings** (amber, may proceed) — altitude > 1 km, SSN > 160 (foF2
  saturated), k0·h > 0.6 (second-order clutter beyond perturbation theory),
  CIT > 300 s, beam > 180°, upper-decile noise above 20 MHz, slow target
  with sea clutter switched off.
Headless mode prints the same messages and exits with code 2 on errors.

| Input | Allowed range |
|---|---|
| Frequency | 1.5–30 MHz (≤ 60 values) |
| Transmit power | 1 W – 10 MW |
| Duty cycle | 0.001–1 |
| Gain | −20 to +30 dBi |
| CIT | 0.1–3600 s |
| Bandwidth | 1–1000 kHz |
| Beamwidth | 1–360° |
| Losses, NF | 0–40 dB |
| Doppler floor | −150 to −20 dB |
| Pd | 0.01–0.999 |
| Pfa | 1e-12 – 0.1 |
| RCS | 1e-4 – 1e6 m² |
| Altitude | 0–3000 m |
| Radial speed (and sweep limits) | −500 to +500 m/s; step 0.01–50 m/s |
| Land section | 0–20 km |
| Earth-radius factor | 0.5–4 |
| Range grid | min 1–100 km, max 20–2000 km, step 0.1–10 km |
| Sea state | 0–7 (integer) |
| Month / UTC hour / SSN | 1–12 / 0–23 / 0–300 |
| Skywave & ionospheric parameters | see `LIMITS` in `hfswr_model.py` |

## Key default assumptions
| Parameter | Default | Note |
|---|---|---|
| Receive beamwidth | 180° | Single element as briefed — clutter-limited |
| Waveform bandwidth | 25 kHz | 6 km range cell |
| Target radial speed | 50 m/s | Doppler-clear of sea echo; set < 15 m/s to see Bragg effects |
| Sea state | 3 | Wind from 225° (SW) |
| Doppler dynamic-range floor | −85 dB | |
| Skywave interference excess | +10 dB | Replace with spectrum-survey data |
| Antenna gain to ionosphere | −15 dB | Ideal monopole has a zenith null; real arrays do not |

## Not modelled
* Barrick sea-roughness propagation loss (needs the full effective-impedance
  integral — next candidate).
* Sky-wave-propagated sea backscatter beyond the skip distance, sporadic-E and
  auroral clutter.
* Swell (bimodal seas), fetch- or duration-limited seas, currents beyond a
  single line-broadening parameter.
* Frequency-dependent RCS, array gain, beam-scan losses.
