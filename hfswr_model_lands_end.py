"""
hfswr_model.py
==============
Physics engine for the HF Surface Wave Radar (HFSWR) performance tool.

Model chain (one-way propagation applied twice, i.e. monostatic):

  1. Ground-wave attenuation factor |W(d, h)|  - ITU-R P.368 method
       * Spherical, smooth, homogeneous earth (effective radius k = 4/3)
       * Surface impedance boundary (vertical polarisation), ground constants
         per ITU-R P.527 (sea of average salinity: sigma = 5 S/m, eps_r = 70)
       * Residue (Fock/Wait) series beyond the near zone, Sommerfeld-Norton
         flat-earth solution inside it, exact Airy height-gain functions
       * Mixed land/sea path by Millington's method (P.368 Sect. 3)
     Normalisation is the P.368 one: |W| = 1 gives 300 mV/m at 1 km for 1 kW
     from a short vertical monopole on a perfect plane (109.54 dBuV/m).

  2. External noise - ITU-R P.372
       * Atmospheric noise: P.372 numerical coefficients (Fam at 1 MHz from
         the Fourier-series maps, frequency dependence, Du/Dl deciles),
         interpolated between 4-hour time blocks exactly as ITU P372.dll
       * Galactic noise 52 - 23 log f, screened when f <= overhead foF2
         (foF2 from the ITU-R P.1239 maps used in P.533, linear in R12,
         R12 capped at 160 as P.1239 requires)
       * Man-made noise categories (city ... quiet rural)
       * Lognormal combination of the three components (P.372 Sect. 8)

  3. Sea clutter (selectable sea state 0-7)
       * Pierson-Moskowitz fully developed sea, cos^4(θ/2) spreading,
         wind direction selectable; Hs from the WMO sea-state code
       * First-order Bragg lines: sigma0 = 2^6 π k0^4 S(2k0) (Barrick 1972),
         at Doppler ±sqrt(g k0/π), averaged over the receive beam
       * Second-order continuum: energy (k0 h)^2/2 relative to first order,
         spread around each line with the dominant ocean-wave frequency
       * Clutter cell = R x beamwidth x c/2B; Doppler processing floor
         (sidelobes/dynamic range) leaks total clutter into every bin

  4. Skywave effects
       * External skywave interference: added when f < MUF(3000)F2 (P.1239
         foF2 x M(3000)F2 at the site), reduced by daytime D-layer absorption
       * Ionospheric self-clutter: own signal at near-vertical incidence when
         f < foF2, at ranges hmF2 ... hmF2 + extent, Doppler-spread about 0 Hz

  5. Radar equation (energy form, coherent integration) and detection:

        SINR = E_target / (N0 + C_sea(fd) + C_iono(fd))

     E_target = Pt·D·T·G²·λ²·σ·|W|⁴ / ((4π)³ R⁴ L);  N0 = kT0(fa + fi + fr - 1)
     Pd from Marcum (Swerling 0) or Swerling 1 with the selected Pfa.
     Detection range = first range at which Pd falls below the required Pd.

Reference implementations checked against:
  * GRWAVE (ITU-R P.368 program, Rotheram 1985) - agreement typically
    within ~0.3 dB over sea 5-30 MHz, 5-400 km (see validate() below)
  * ITU P372.dll (ITU-R-HF repository) - atmospheric noise reproduced to
    better than 0.01 dB at Land's End
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict, replace

import numpy as np
from scipy.special import airye, ai_zeros, wofz

C0 = 299_792_458.0
K_BOLTZ = 1.380649e-23
T0 = 290.0
EARTH_RADIUS = 6_371_000.0
E_REF_DBUV = 109.54  # P.368 reference: 300 mV/m @ 1 km, 1 kW, short monopole

# P.527 / P.368 ground types (sigma S/m, eps_r)
GROUND_TYPES = {
    "Sea, average salinity (5 S/m, 70)": (5.0, 70.0),
    "Sea, low salinity (1 S/m, 80)": (1.0, 80.0),
    "Wet ground (10 mS/m, 30)": (0.01, 30.0),
    "Land (30 mS/m, 40)": (0.03, 40.0),
    "Medium dry ground (1 mS/m, 15)": (0.001, 15.0),
    "Very dry ground (0.1 mS/m, 3)": (0.0001, 3.0),
}

# P.372 man-made noise: Fam = c - d log10(f_MHz); (c, d, Du, Dl)
MAN_MADE = {
    "City": (76.8, 27.7, 11.0, 6.7),
    "Residential": (72.5, 27.7, 10.6, 5.3),
    "Rural": (67.2, 27.7, 9.2, 4.6),
    "Quiet rural": (53.6, 28.6, 9.2, 4.6),
}

_E23 = np.exp(-2j * np.pi / 3)


# =============================================================================
# 1. Ground-wave propagation (ITU-R P.368)
# =============================================================================
def _airy_ratio(t):
    """w'(t)/w(t) for the Fock function w(t) ∝ Ai(t·e^{-2πi/3})."""
    z = t * _E23
    a, ap, _, _ = airye(z)
    return _E23 * ap / a


from functools import lru_cache


@lru_cache(maxsize=256)
def _modal_roots_cached(q_re, q_im, n):
    return _modal_roots(complex(q_re, q_im), n)


def _modal_roots(q, n, steps=None):
    """Roots t_s of w'(t) - q w(t) = 0 by continuation in q from q = 0."""
    _, ap, _, _ = ai_zeros(n)
    t = -ap * np.exp(-1j * np.pi / 3)          # q = 0 (perfect conductor)
    if steps is None:
        steps = int(np.clip(1.5 * abs(q), 20, 200))
    dq = q / steps
    qq = 0.0
    for _ in range(steps):
        t = t + dq / (t - qq * qq)                 # Euler predictor: dt/dq = 1/(t - q^2)
        qq = qq + dq
        for _ in range(40):                        # Newton corrector
            r = _airy_ratio(t)
            dt = (r - qq) / (t - r * r)
            t = t - dt
            if np.max(np.abs(dt) / np.abs(t)) < 1e-12:
                break
    return t


def _height_gain(t, y):
    """Fock height-gain functions w(t - y)/w(t), overflow-safe."""
    if y == 0:
        return np.ones_like(t)
    z1, z0 = (t - y) * _E23, t * _E23
    return (airye(z1)[0] / airye(z0)[0]) * np.exp(-2.0 / 3.0 * (z1 ** 1.5 - z0 ** 1.5))


def ground_wave_W(f_mhz, d_m, sigma, eps_r, h_m=0.0, k_factor=4.0 / 3.0):
    """
    Complex-magnitude attenuation factor |W| (linear) relative to a perfectly
    conducting flat earth, for a ground-based vertical source and an observer
    at height h_m. d_m is an array of ground ranges (m).
    """
    d = np.atleast_1d(np.asarray(d_m, dtype=float))
    lam = C0 / (f_mhz * 1e6)
    k = 2 * np.pi / lam
    a = EARTH_RADIUS * k_factor
    eta = eps_r - 1j * 60.0 * lam * sigma          # e^{+jωt} convention
    delta = np.sqrt(eta - 1.0) / eta               # normalised surface impedance (V-pol)
    m = (k * a / 2.0) ** (1.0 / 3.0)
    q = -1j * m * delta
    x = m * d / a
    y = k * h_m / m

    out = np.empty_like(d)

    # --- near zone: Sommerfeld-Norton flat earth with height gain ---
    x_sw = 0.08
    near = x < x_sw
    if np.any(near):
        w = -1j * k * d[near] / 2.0 * delta ** 2
        F = 1.0 - 1j * np.sqrt(np.pi * w) * wofz(-np.sqrt(w))
        out[near] = np.abs(F * (1.0 + 1j * k * h_m * delta))

    # --- far zone: residue series ---
    far = ~near
    if np.any(far):
        n = int(np.clip(math.ceil(((35.0 / (0.866 * x_sw)) ** 1.5) / 4.71), 60, 4000))
        ts = _modal_roots_cached(round(q.real, 12), round(q.imag, 12), n)
        hg = _height_gain(ts, y)
        coef = hg / (ts - q * q)
        xf = x[far][:, None]
        S = np.sum(np.exp(-1j * xf * ts[None, :]) * coef[None, :], axis=1)
        out[far] = np.sqrt(np.pi * x[far]) * np.abs(S)
    return out


def field_strength_dbuv(f_mhz, d_m, sigma, eps_r, h_m=0.0, p_kw=1.0, k_factor=4 / 3):
    """P.368-style field strength (dBuV/m) for p_kw from a short monopole."""
    d = np.atleast_1d(d_m)
    W = ground_wave_W(f_mhz, d, sigma, eps_r, h_m, k_factor)
    return E_REF_DBUV + 10 * np.log10(p_kw) - 20 * np.log10(d / 1e3) + 20 * np.log10(W)


def mixed_path_W_db(f_mhz, d_m, land, sea, d_land_m, h_m=0.0, k_factor=4 / 3):
    """
    20log10|W| for a two-section path: land (length d_land_m, at the radar)
    followed by sea, by Millington's method. Target height gain is applied
    for the sea section at the far end.
    """
    d = np.atleast_1d(np.asarray(d_m, dtype=float))
    WS = lambda r, h=0.0: 20 * np.log10(ground_wave_W(f_mhz, r, *sea, h, k_factor))
    if d_land_m <= 0:
        return WS(d, h_m)
    WL = lambda r: 20 * np.log10(ground_wave_W(f_mhz, r, *land, 0.0, k_factor))
    # Field (dB) = W_dB - 20log d ; the -20log d terms cancel in Millington
    d1 = d_land_m
    dd = np.maximum(d, d1 + 1.0)
    d2 = dd - d1
    wl_d1, ws_d1 = WL(np.array([d1]))[0], WS(np.array([d1]))[0]
    # 20log d differences must be retained where distances differ
    E = lambda wdb, r: wdb - 20 * np.log10(r)
    Ef = E(wl_d1, d1) - E(ws_d1, d1) + E(WS(dd), dd)
    Er = E(WS(d2), d2) - E(WL(d2), d2) + E(WL(dd), dd)
    Wmix = 0.5 * (Ef + Er) + 20 * np.log10(dd)
    hgain = WS(dd, h_m) - WS(dd) if h_m > 0 else 0.0
    res = Wmix + hgain
    res[d <= d1] = WL(d[d <= d1])  # target over land (not used in practice)
    return res


# =============================================================================
# 2. Noise (ITU-R P.372)
# =============================================================================
ATMOS_MODELS = ("ITU-R P.372 tables", "Parametric (approx.)")

# -----------------------------------------------------------------------------
# Parametric atmospheric-noise model
# -----------------------------------------------------------------------------
# Closed form with the same structure as P.372 (noise at 1 MHz, then a family
# of frequency curves linear in that level):
#
#   Fam1(month, LMT) = Σ c_ij · T_i(LMT) · S_j(month)
#        T = [1, cos t, sin t, cos 2t, sin 2t, cos 3t, sin 3t],  t = 2π·LMT/24
#        S = [1, cos s, sin s],                                  s = 2π(month-1)/12
#   FaA(f) = α(x)·Fam1 + β(x),  x = log10(f MHz), α and β cubic in x
#   Du, Dl = d0 + d1 x + d2 x² + d3 cos t + d4 sin t
#
# Coefficients: least-squares fit to the ITU-R P.372 tables at Land's End
# (all months, all hours, 3-30 MHz), weighted by how much atmospheric noise
# matters against the quietest man-made + galactic floor. Accuracy against
# P.372, in total median noise: 0.4 dB rms (Rural), 1.3 dB rms / 2.8 dB p95
# (Quiet rural). FaA alone is less accurate (≈4.6 dB rms, worst above
# 20 MHz where it is far below other noise). Refit for another site with
# fit_parametric_atmos().
PARAMETRIC_ATMOS = {
    "fitted_to": "ITU-R P.372 tables, Land's End (50.066 N, 5.715 W), 3-30 MHz",
    "lat": 50.066, "lon": -5.715, "f_min": 3.0, "f_max": 30.0,
    "fam": [33.557099, -1.901051, 1.147869, -25.449913, -2.060462, -0.652544, 11.314784, -1.291786, 0.332131, 4.184948, 3.793512, -1.635571, -4.573687, -2.973094, 1.423109, -0.203472, -0.211851, -0.145886, 2.129503, 3.236088, -1.858677],
    "alpha": [1.882268, -9.388142, 11.48071, -4.075193],
    "beta": [52.431601, 45.030667, -53.153673, -4.365492],
    "du": [17.375686, -13.646395, 3.636936, -1.553489, 0.196162],
    "dl": [14.085948, -9.433446, 1.512531, -0.867178, 0.084836],
}


def _param_basis(month, lmt):
    t = 2 * np.pi * np.asarray(lmt, dtype=float) / 24.0
    s = 2 * np.pi * (np.asarray(month, dtype=float) - 1) / 12.0
    T = [np.ones_like(t), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t),
         np.cos(3 * t), np.sin(3 * t)]
    S = [np.ones_like(s), np.cos(s), np.sin(s)]
    return np.array([a * b for a in T for b in S])


def _param_eval(c, month, lmt, f):
    B = _param_basis(month, lmt)
    x = np.log10(np.asarray(f, dtype=float))
    fam1 = np.asarray(c["fam"]) @ B
    al = sum(k * x ** i for i, k in enumerate(c["alpha"]))
    be = sum(k * x ** i for i, k in enumerate(c["beta"]))
    t = 2 * np.pi * np.asarray(lmt, dtype=float) / 24.0
    dec = lambda d: d[0] + d[1] * x + d[2] * x ** 2 + d[3] * np.cos(t) + d[4] * np.sin(t)
    return al * fam1 + be, dec(c["du"]), dec(c["dl"])


def parametric_atmospheric(month, lmt_hour, f, coeffs=None):
    """Parametric FaA, DuA, DlA (dB) - see PARAMETRIC_ATMOS."""
    fa, du, dl = _param_eval(coeffs or PARAMETRIC_ATMOS, month, lmt_hour, f)
    return float(fa), float(max(du, 0.5)), float(max(dl, 0.5))


def fit_parametric_atmos(site_noise, f_min=3.0, f_max=30.0):
    """
    Refit the parametric model to the P.372 tables of another site
    (a SiteNoise object). Returns (coeffs, stats). Assign the coeffs to
    PARAMETRIC_ATMOS (or pass them to parametric_atmospheric) to use them.
    """
    from scipy.optimize import least_squares
    rows = [(m, h, f, *site_noise.atmospheric(m, h, f))
            for m in range(1, 13) for h in range(24) for f in np.arange(f_min, f_max + 1e-9, 0.5)]
    mo, h, f, Fa, Du, Dl = np.array(rows).T
    B = _param_basis(mo, h)
    x = np.log10(f)
    nb = B.shape[0]

    def model(c):
        al = c[nb] + c[nb + 1] * x + c[nb + 2] * x ** 2 + c[nb + 3] * x ** 3
        be = c[nb + 4] + c[nb + 5] * x + c[nb + 6] * x ** 2 + c[nb + 7] * x ** 3
        return al * (c[:nb] @ B) + be

    c0 = np.zeros(nb + 8); c0[0], c0[nb], c0[nb + 1] = 60.0, 1.0, -0.5
    c = least_squares(lambda c: model(c) - Fa, c0).x
    floor = 10 * np.log10(10 ** ((53.6 - 28.6 * x) / 10) + 10 ** ((52 - 23 * x) / 10))
    wt = np.sqrt(0.05 + 10 ** (Fa / 10) / (10 ** (Fa / 10) + 10 ** (floor / 10)))
    c = least_squares(lambda c: (model(c) - Fa) * wt, c).x
    A = np.vstack([np.ones_like(x), x, x ** 2, np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * h / 24)]).T
    du = np.linalg.lstsq(A, Du, rcond=None)[0]
    dl = np.linalg.lstsq(A, Dl, rcond=None)[0]
    coeffs = {"fitted_to": f"ITU-R P.372 tables, {site_noise.name}, {f_min:g}-{f_max:g} MHz",
              "lat": site_noise.lat, "lon": site_noise.lon, "f_min": f_min, "f_max": f_max,
              "fam": c[:nb].tolist(), "alpha": c[nb:nb + 4].tolist(), "beta": c[nb + 4:].tolist(),
              "du": du.tolist(), "dl": dl.tolist()}
    err = model(c) - Fa
    return coeffs, {"FaA_rms_db": float(np.sqrt(np.mean(err ** 2))),
                    "FaA_max_db": float(np.max(np.abs(err)))}


class SiteNoise:
    """P.372 noise for a fixed receiving site, from build_site_data.py JSON."""

    def __init__(self, path):
        with open(path) as fh:
            self.site = json.load(fh)
        self.name = self.site["name"]
        self.lat, self.lon = self.site["lat"], self.site["lon"]

    # -- atmospheric ---------------------------------------------------------
    def _block(self, month, tb, f):
        M = self.site["months"][month - 1]
        fam1 = M["fam1"][tb]
        fam = M["fam"][tb]
        u = [-0.75, (8.0 * 2.0 ** math.log10(f) - 11.0) / 4.0]
        cz = pz = px = 0.0
        for kk in range(2):
            pz = u[kk] * fam[0] + fam[1]
            px = u[kk] * fam[7] + fam[8]
            for j in range(2, 7):
                pz = u[kk] * pz + fam[j]
                px = u[kk] * px + fam[j + 7]
            if kk == 0:
                cz = fam1 * (2.0 - pz) - px
        FA = cz * pz + px
        x = math.log10(min(f, 20.0))
        v = []
        for i in range(2):  # Du, Dl
            c = M["dud"][i][tb]
            yv = c[0]
            for kk in range(1, 5):
                yv = yv * x + c[kk]
            v.append(yv)
        return FA, v[0], v[1]

    def atmospheric(self, month, lmt_hour, f):
        """FaA, DuA, DlA (dB) - interpolation as ITU P372.dll."""
        h = int(lmt_hour) % 24
        tb = (h // 4) % 6
        tb2 = (tb + 1) % 6
        slp = (h % 4) / 4.0
        a = self._block(month, tb, f)
        b = self._block(month, tb2, f)
        lin = lambda p, r: 10 * math.log10(10 ** (p / 10) + (10 ** (r / 10) - 10 ** (p / 10)) * slp)
        return lin(a[0], b[0]), lin(a[1], b[1]), lin(a[2], b[2])

    # -- ionosphere ---------------------------------------------------------
    def fof2(self, month, utc_hour, ssn):
        M = self.site["months"][month - 1]
        r = min(max(ssn, 0.0), 160.0)
        h = int(utc_hour) % 24
        return (M["fof2_r0"][h] * (100 - r) + M["fof2_r100"][h] * r) / 100.0

    def m3000(self, month, utc_hour, ssn):
        M = self.site["months"][month - 1]
        r = min(max(ssn, 0.0), 160.0)
        h = int(utc_hour) % 24
        return (M["m3k_r0"][h] * (100 - r) + M["m3k_r100"][h] * r) / 100.0

    # -- combined -----------------------------------------------------------
    def total(self, month, utc_hour, ssn, f, man_made="Rural", percentile="Median",
              galactic_screening=True, atmos_model=ATMOS_MODELS[0]):
        """
        Returns dict with FaA, FaG, FaM and combined Fa (dB above kT0b).
        percentile: 'Median' or 'Upper decile' (noise exceeded 10 % of time).
        atmos_model: 'ITU-R P.372 tables' or 'Parametric (approx.)'.
        """
        lmt = (int(utc_hour) + int(self.lon / 15.0)) % 24
        if atmos_model == ATMOS_MODELS[1]:
            FaA, DuA, DlA = parametric_atmospheric(month, lmt, f)
        else:
            FaA, DuA, DlA = self.atmospheric(month, lmt, f)
        c, dd, DuM, DlM = MAN_MADE[man_made]
        FaM = c - dd * math.log10(f)
        FaG = 52.0 - 23.0 * math.log10(f)
        foF2 = self.fof2(month, utc_hour, ssn)
        g_on = (f > foF2) or not galactic_screening
        comps = [(FaA, DuA, DlA), (FaM, DuM, DlM)]
        if g_on:
            comps.append((FaG, 2.0, 2.0))
        Fam, Du, _ = _combine(comps)
        Fa = Fam + (Du if percentile.startswith("Upper") else 0.0)
        return {"FaA": FaA, "FaM": FaM, "FaG": FaG if g_on else -np.inf,
                "foF2": foF2, "Fa": Fa}


def _combine(comps):
    """P.372 Sect. 8 lognormal combination (as ITU P372.dll)."""
    c = 10.0 / math.log(10.0)
    res = []
    for which in (1, 2):  # upper, lower
        sig = [(cp[which] / 1.282) for cp in comps]
        alpha = sum(math.exp(cp[0] / c + s * s / (2 * c * c)) for cp, s in zip(comps, sig))
        beta = sum(math.exp(cp[0] / c + s * s / (2 * c * c)) ** 2 * (math.exp((s / c) ** 2) - 1)
                   for cp, s in zip(comps, sig))
        gamma = sum(math.exp(cp[0] / c) for cp in comps)
        if any(cp[which] > 12.0 for cp in comps):
            sT = c * math.sqrt(2.0 * math.log(alpha / gamma))
        else:
            sT = c * math.sqrt(math.log(1.0 + beta / alpha ** 2))
        res.append((c * (math.log(alpha) - sT * sT / (2 * c * c)), 1.282 * sT))
    return min(res[0][0], res[1][0]), res[0][1], res[1][1]



# =============================================================================
# 3. Ionosphere helpers (P.1239 maps already in the site file)
# =============================================================================
def solar_zenith_deg(lat_deg, lon_deg, month, utc_hour):
    """Solar zenith angle at mid-month (simple declination / hour-angle model)."""
    doy = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349][month - 1]
    dec = math.radians(23.44) * math.sin(math.radians(360.0 / 365.0 * (284 + doy)))
    lst = (utc_hour + lon_deg / 15.0) % 24.0
    ha = math.radians(15.0 * (lst - 12.0))
    lat = math.radians(lat_deg)
    cz = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
    return math.degrees(math.acos(max(-1.0, min(1.0, cz))))


def d_layer_absorption_db(f_mhz, chi_deg, ssn):
    """
    Approximate one-way vertical-incidence non-deviative absorption (dB),
    George/Bradley form: 430 (1 + 0.0037 R) cos^1.3(0.881 chi) / (f + fH)^2,
    fH = 1.4 MHz. Zero at night. Engineering approximation, not P.533.
    """
    x = 0.881 * chi_deg
    if x >= 90.0:
        return 0.0
    r = min(max(ssn, 0.0), 160.0)
    return 430.0 * (1 + 0.0037 * r) * math.cos(math.radians(x)) ** 1.3 / (f_mhz + 1.4) ** 2


# =============================================================================
# 4. Sea clutter (Barrick first-order Bragg + second-order continuum)
# =============================================================================
G_ACC = 9.81

# WMO / Douglas sea state -> representative significant wave height (m)
SEA_STATES = {
    0: ("Calm (glassy)", 0.02),
    1: ("Calm (rippled)", 0.05),
    2: ("Smooth", 0.30),
    3: ("Slight", 0.875),
    4: ("Moderate", 1.875),
    5: ("Rough", 3.25),
    6: ("Very rough", 5.0),
    7: ("High", 7.5),
}


def sea_state_wind(ss):
    """Wind speed U19.5 (m/s) for a fully developed PM sea with the SS's Hs."""
    hs = SEA_STATES[int(ss)][1]
    return math.sqrt(hs * G_ACC / 0.21), hs


def pm_directional(k, theta, u19, s=2):
    """
    Pierson-Moskowitz directional wave-height spectrum S(k, theta) (m^4),
    normalised so that  ∫∫ S k dk dθ = <ζ²>. theta is measured from the
    direction in which the wind blows (wave propagation), cos^2s(θ/2) spread.
    """
    alpha, beta = 0.0081, 0.74
    F = alpha / (2.0 * k ** 3) * np.exp(-beta * G_ACC ** 2 / (k ** 2 * u19 ** 4))
    norm = 2 ** (2 * s) * math.gamma(s + 1) ** 2 / (2 * math.pi * math.gamma(2 * s + 1))
    G = norm * np.cos(np.asarray(theta) / 2.0) ** (2 * s)
    return F / k * G


def bragg_frequency_hz(f_mhz):
    k0 = 2 * math.pi * f_mhz * 1e6 / C0
    return math.sqrt(2.0 * G_ACC * k0) / (2 * math.pi)


def sea_clutter_spectrum(f_mhz, sea_state, wind_from_deg, look_az_deg, beamwidth_deg):
    """
    Returns a dict describing the normalised sea-echo Doppler spectrum:
      sigma1_pos, sigma1_neg : first-order sigma0 (m²/m²) of the +/- Bragg lines
      sigma2                 : total second-order sigma0 (m²/m²)
      fB                     : Bragg Doppler (Hz)
      fp                     : dominant ocean-wave frequency (Hz), sets the
                               second-order continuum width
    First order: sigma0 = 2^6 π k0^4 S(2k0) (Barrick 1972), averaged over the
    receive beam. Second order: total energy ratio (k0 h)^2 / 2 relative to
    first order (from Barrick's wave-height inversion relation with unit
    weighting), capped at 0.5 where perturbation theory breaks down.
    """
    k0 = 2 * math.pi * f_mhz * 1e6 / C0
    u19, hs = sea_state_wind(sea_state)
    wave_dir = math.radians((wind_from_deg + 180.0) % 360.0)   # waves travel downwind
    bw = max(beamwidth_deg, 1e-3)
    az = np.radians(look_az_deg + np.linspace(-bw / 2, bw / 2, 61))
    # +ve Doppler: waves travelling towards the radar (direction az + 180°)
    th_pos = (az + math.pi) - wave_dir
    th_neg = az - wave_dir
    kb = 2 * k0
    c1 = 2 ** 6 * math.pi * k0 ** 4
    s_pos = float(np.mean(c1 * pm_directional(kb, th_pos, u19)))
    s_neg = float(np.mean(c1 * pm_directional(kb, th_neg, u19)))
    h_rms = hs / 4.0
    r2 = min((k0 * h_rms) ** 2 / 2.0, 0.5)
    fp = 0.877 * G_ACC / u19 / (2 * math.pi)
    return {"sigma1_pos": s_pos, "sigma1_neg": s_neg, "sigma2": r2 * (s_pos + s_neg),
            "fB": bragg_frequency_hz(f_mhz), "fp": fp, "u19": u19, "hs": hs,
            "k0h": k0 * h_rms}


def sea_clutter_psd(fd, spec, line_sd_hz):
    """
    Doppler power-spectral shape (1/Hz, per unit total sigma0) evaluated at
    target Doppler fd. First-order lines are Gaussians of s.d. line_sd_hz;
    the second-order continuum around each line is (1 + (Δf/fp)^2)^-5/2.
    Returns (psd_first, psd_second) each already weighted by sigma0 (m²/m²/Hz).
    """
    fB, fp = spec["fB"], max(spec["fp"], 1e-3)
    g = lambda x: np.exp(-0.5 * (x / line_sd_hz) ** 2) / (line_sd_hz * math.sqrt(2 * math.pi))
    first = spec["sigma1_pos"] * g(fd - fB) + spec["sigma1_neg"] * g(fd + fB)
    s1 = spec["sigma1_pos"] + spec["sigma1_neg"]
    if s1 <= 0:
        return first, 0.0 * first
    w_pos, w_neg = spec["sigma1_pos"] / s1, spec["sigma1_neg"] / s1
    # (1 + x²)^-5/2 integrates to 4/3; the far tail falls as Δf^-5, matching the
    # ω^-5 high-frequency tail of the PM spectrum that feeds second-order echo.
    lor = lambda x: (0.75 / fp) / (1 + (x / fp) ** 2) ** 2.5
    second = spec["sigma2"] * (w_pos * lor(fd - fB) + w_neg * lor(fd + fB))
    return first, second


# =============================================================================
# 5. Detection statistics
# =============================================================================
def pd_from_snr(snr_lin, pfa, swerling=0):
    """Single-look Pd after coherent integration. Swerling 0 (Marcum) or 1."""
    from scipy.stats import ncx2
    snr = np.maximum(np.asarray(snr_lin, dtype=float), 0.0)
    thr = -2.0 * math.log(pfa)
    if int(swerling) == 1:
        return pfa ** (1.0 / (1.0 + snr))
    return ncx2.sf(thr, 2, 2.0 * snr)


def snr_required_db(pd, pfa, swerling=0):
    """Inverse of pd_from_snr (dB)."""
    from scipy.optimize import brentq
    f = lambda x: pd_from_snr(10 ** (x / 10), pfa, swerling) - pd
    return brentq(f, -10.0, 60.0)


# =============================================================================
# 6. Radar model
# =============================================================================
@dataclass
class RadarParams:
    # --- radar ---
    pt_w: float = 10e3                 # transmit power (W, average if duty < 1)
    duty: float = 1.0                  # duty cycle (FMCW = 1.0)
    gain_dbi: float = 5.16             # Tx and Rx gain incl. ideal ground plane
    cit_s: float = 10.0                # coherent integration time
    bandwidth_hz: float = 25e3         # waveform bandwidth -> range cell c/2B
    rx_beamwidth_deg: float = 180.0    # azimuth extent of the clutter cell
    look_az_deg: float = 180.0         # boresight (due south)
    losses_db: float = 6.0
    rx_nf_db: float = 10.0
    doppler_dr_db: float = -85.0       # Doppler sidelobe / dynamic-range floor
    # --- detection ---
    pd_req: float = 0.9
    pfa: float = 1e-6
    swerling: int = 0
    # --- target ---
    rcs_m2: float = 1.0
    target_h_m: float = 50.0
    target_vr_ms: float = 50.0         # radial speed (sets Doppler)
    # --- propagation path ---
    sea: str = "Sea, average salinity (5 S/m, 70)"
    land: str = "Medium dry ground (1 mS/m, 15)"
    land_km: float = 0.5
    k_factor: float = 1.355            # matches GRWAVE (Ns=315, exponential atmosphere)
    r_min_km: float = 5.0
    r_max_km: float = 800.0
    r_step_km: float = 1.0
    # --- noise ---
    man_made: str = "Rural"
    percentile: str = "Median"
    atmos_model: str = "ITU-R P.372 tables"   # or "Parametric (approx.)"
    galactic_screening: bool = True
    # --- sea clutter ---
    sea_clutter: bool = True
    sea_state: int = 3
    wind_from_deg: float = 225.0       # prevailing SW
    bragg_sd_hz: float = 0.02          # first-order line broadening (currents etc.)
    # --- skywave ---
    sky_interference: bool = True
    sky_interf_excess_db: float = 10.0  # interference above P.372 noise when band open
    iono_clutter: bool = True
    zenith_gain_db: float = -15.0      # Tx/Rx gain towards ionosphere rel. to horizon
    iono_refl_loss_db: float = 6.0     # F-layer reflection / scattering loss
    iono_doppler_sd_hz: float = 0.5    # ionospheric clutter Doppler spread (s.d.)
    iono_extent_km: float = 150.0      # range extent of self-clutter beyond h'F
    # --- radial-speed sweep (fourth chart) ---
    speed_min_ms: float = 0.0
    speed_max_ms: float = 60.0
    speed_step_ms: float = 0.25
    freqs_mhz: list = field(default_factory=lambda: list(range(5, 31)))

    @property
    def snr_req_db(self):
        return snr_required_db(self.pd_req, self.pfa, self.swerling)


TOOL_VERSION = "3.2"

# Every RadarParams field: (section, unit, description). Used for the CSV
# parameter record; parameter_record() refuses to run if a field is missing.
PARAM_META = {
    "pt_w":                 ("Radar", "W", "Transmit power (average if duty < 1)"),
    "duty":                 ("Radar", "", "Transmit duty cycle"),
    "gain_dbi":             ("Radar", "dBi", "Tx and Rx antenna gain incl. ideal ground-plane image"),
    "cit_s":                ("Radar", "s", "Coherent integration time"),
    "bandwidth_hz":         ("Radar", "Hz", "Waveform bandwidth"),
    "rx_beamwidth_deg":     ("Radar", "deg", "Receive azimuth beamwidth (clutter cell width)"),
    "look_az_deg":          ("Radar", "deg true", "Boresight azimuth"),
    "losses_db":            ("Radar", "dB", "System and processing losses"),
    "rx_nf_db":             ("Radar", "dB", "Receiver noise figure"),
    "doppler_dr_db":        ("Radar", "dB", "Doppler sidelobe / dynamic-range floor"),
    "pd_req":               ("Detection", "", "Required probability of detection"),
    "pfa":                  ("Detection", "", "Probability of false alarm"),
    "swerling":             ("Detection", "", "Target fluctuation model (0 = steady, 1 = Swerling 1)"),
    "rcs_m2":               ("Target", "m²", "Radar cross-section"),
    "target_h_m":           ("Target", "m", "Target altitude"),
    "target_vr_ms":         ("Target", "m/s", "Target radial speed for charts 1-3 (+ve closing)"),
    "sea":                  ("Path", "", "Sea ground type (P.527)"),
    "land":                 ("Path", "", "Land ground type between radar and shore"),
    "land_km":              ("Path", "km", "Land section length (Millington)"),
    "k_factor":             ("Path", "", "Effective earth-radius factor"),
    "r_min_km":             ("Path", "km", "Minimum range computed"),
    "r_max_km":             ("Path", "km", "Maximum range computed"),
    "r_step_km":            ("Path", "km", "Range step"),
    "man_made":             ("Noise", "", "P.372 man-made noise category"),
    "percentile":           ("Noise", "", "Noise statistic (median or upper decile)"),
    "atmos_model":          ("Noise", "", "Atmospheric noise: ITU-R P.372 tables or parametric approximation"),
    "galactic_screening":   ("Noise", "", "Galactic noise screened when f <= foF2"),
    "sea_clutter":          ("Sea clutter", "", "Sea clutter included"),
    "sea_state":            ("Sea clutter", "WMO code", "Sea state"),
    "wind_from_deg":        ("Sea clutter", "deg true", "Wind direction (from)"),
    "bragg_sd_hz":          ("Sea clutter", "Hz", "First-order Bragg line broadening (s.d.)"),
    "sky_interference":     ("Skywave", "", "External skywave interference included"),
    "sky_interf_excess_db": ("Skywave", "dB", "Interference above P.372 median noise when band open"),
    "iono_clutter":         ("Skywave", "", "Ionospheric self-clutter included"),
    "zenith_gain_db":       ("Skywave", "dB", "Tx/Rx gain towards ionosphere relative to boresight"),
    "iono_refl_loss_db":    ("Skywave", "dB", "F-layer reflection / scattering loss"),
    "iono_doppler_sd_hz":   ("Skywave", "Hz", "Ionospheric clutter Doppler spread (s.d.)"),
    "iono_extent_km":       ("Skywave", "km", "Self-clutter range extent beyond hmF2"),
    "speed_min_ms":         ("Speed sweep", "m/s", "Chart 4 minimum radial speed"),
    "speed_max_ms":         ("Speed sweep", "m/s", "Chart 4 maximum radial speed"),
    "speed_step_ms":        ("Speed sweep", "m/s", "Chart 4 speed step"),
    "freqs_mhz":            ("Frequencies", "MHz", "Frequencies plotted in range mode"),
}


def parameter_record(model, p, mode, freq, month, hour, ssn):
    """
    Complete, self-describing record of a run as rows of
    (section, parameter, value, unit, description): run metadata, display
    settings, every RadarParams field, model constants and derived values
    (scenario-wide and per frequency at the fixed month/hour/SSN).
    """
    import datetime
    missing = set(p.__dataclass_fields__) - set(PARAM_META)
    if missing:
        raise RuntimeError(f"PARAM_META lacks {sorted(missing)} - update hfswr_model.py")
    rows = []
    add = lambda sec, name, val, unit="", desc="": rows.append((sec, name, val, unit, desc))
    g = lambda v, n=6: f"{v:.{n}g}" if isinstance(v, float) else v

    # run metadata
    add("Run", "tool_version", TOOL_VERSION)
    add("Run", "generated_utc", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    add("Run", "site_name", model.noise.name, "", "Receiver site (noise and ionosphere data)")
    add("Run", "site_lat", g(model.noise.lat), "deg N")
    add("Run", "site_lon", g(model.noise.lon), "deg E")
    # display / chart settings
    add("Display", "mode", mode, "", {"range": "Maximum detection range", "pdmap": "Pd map",
                                      "pdcurves": "Pd curves"}.get(mode, ""))
    if mode != "range":
        add("Display", "pd_frequency_mhz", g(float(freq)), "MHz", "Frequency for Pd displays")
    add("Display", "fixed_month", month, "", "Month for diurnal, solar-cycle and speed charts")
    add("Display", "fixed_utc_hour", hour, "UTC", "Hour for annual, solar-cycle and speed charts")
    add("Display", "fixed_ssn", g(float(ssn)), "R12", "Sunspot number for diurnal, annual and speed charts")
    # every input parameter
    for name in p.__dataclass_fields__:
        sec, unit, desc = PARAM_META[name]
        val = getattr(p, name)
        if isinstance(val, (list, tuple)):
            val = " ".join(f"{v:g}" for v in val)
        elif isinstance(val, bool):
            val = "yes" if val else "no"
        add(sec, name, g(val), unit, desc)

    # model constants
    sig, er = GROUND_TYPES[p.sea]
    add("Constants", "sea_conductivity", g(sig), "S/m", "From sea ground type")
    add("Constants", "sea_permittivity", g(er), "", "Relative permittivity")
    sig, er = GROUND_TYPES[p.land]
    add("Constants", "land_conductivity", g(sig), "S/m", "From land ground type")
    add("Constants", "land_permittivity", g(er), "", "Relative permittivity")
    c, d, du, dl = MAN_MADE[p.man_made]
    add("Constants", "man_made_c", g(c), "dB", "P.372 Fam = c - d log10(f MHz)")
    add("Constants", "man_made_d", g(d), "dB/decade")
    add("Constants", "man_made_Du_Dl", f"{du:g} / {dl:g}", "dB", "Upper / lower decile deviations")
    add("Constants", "galactic_noise", "52 - 23 log10(f)", "dB", "P.372 galactic noise")
    if p.atmos_model == ATMOS_MODELS[1]:
        pa = PARAMETRIC_ATMOS
        add("Constants", "parametric_atmos_fitted_to", pa["fitted_to"], "", "Source of parametric coefficients")
        for key in ("fam", "alpha", "beta", "du", "dl"):
            add("Constants", f"parametric_atmos_{key}", " ".join(f"{v:.6g}" for v in pa[key]), "",
                "Parametric atmospheric-noise coefficients")
    else:
        add("Constants", "atmos_noise_source", "ITU-R P.372 numerical coefficients (ITU-R SG3 P372 data)",
            "", f"Site file for {model.noise.name}")
    add("Constants", "earth_radius", g(EARTH_RADIUS), "m", "Before k-factor")
    add("Constants", "T0", g(T0), "K", "Reference temperature")

    # scenario-wide derived values
    add("Derived", "required_sinr", f"{p.snr_req_db:.2f}", "dB", "From Pd, Pfa and Swerling model")
    add("Derived", "range_cell", f"{C0 / (2 * p.bandwidth_hz) / 1e3:.3f}", "km", "c / 2B")
    add("Derived", "doppler_bin", g(1.0 / p.cit_s), "Hz", "1 / CIT")
    u19, hs = sea_state_wind(p.sea_state)
    add("Derived", "sea_state_name", SEA_STATES[int(p.sea_state)][0])
    add("Derived", "significant_wave_height", g(hs), "m", "WMO sea-state code (mid-band)")
    add("Derived", "wind_speed_U19.5", f"{u19:.2f}", "m/s", "Fully developed Pierson-Moskowitz sea")
    add("Derived", "dominant_wave_frequency", f"{0.877 * G_ACC / u19 / (2 * math.pi):.4f}", "Hz")
    sky = model.skywave_state(month, hour, ssn, 10.0)
    add("Derived", "foF2", f"{sky['foF2']:.3f}", "MHz", "At fixed month/hour/SSN (P.1239)")
    add("Derived", "M3000F2", f"{sky['M3000']:.4f}", "", "At fixed month/hour/SSN (P.1239)")
    add("Derived", "MUF3000F2", f"{sky['MUF3000']:.3f}", "MHz", "foF2 x M(3000)F2")
    add("Derived", "hmF2", f"{sky['hF_km']:.1f}", "km", "Shimazaki, from M(3000)F2")
    add("Derived", "solar_zenith_angle", f"{sky['chi']:.2f}", "deg", "At fixed month/hour")

    # per-frequency derived values at the fixed month/hour/SSN
    freqs = list(p.freqs_mhz) if mode == "range" else [freq]
    for f in freqs:
        lam = C0 / (f * 1e6)
        fb = bragg_frequency_hz(f)
        sp = sea_clutter_spectrum(f, p.sea_state, p.wind_from_deg, p.look_az_deg, p.rx_beamwidth_deg)
        s1 = sp["sigma1_pos"] + sp["sigma1_neg"]
        nz = model.noise.total(month, hour, ssn, f, p.man_made, p.percentile, p.galactic_screening,
                               p.atmos_model)
        sk = model.skywave_state(month, hour, ssn, f)
        db = lambda v: f"{10 * math.log10(v):.2f}" if v > 0 else "-inf"
        sec = f"Per frequency {f:g} MHz"
        add(sec, "wavelength", f"{lam:.3f}", "m")
        add(sec, "target_doppler", f"{2 * p.target_vr_ms / lam:.4f}", "Hz", "At charts 1-3 radial speed")
        add(sec, "bragg_frequency", f"{fb:.4f}", "Hz", "First-order Bragg line offset")
        add(sec, "bragg_speed", f"{fb * lam / 2:.3f}", "m/s", "Radial speed with Bragg Doppler")
        add(sec, "sigma0_first_order", db(s1), "dB m²/m²", "Both lines, beam-averaged")
        add(sec, "sigma0_second_order", db(sp["sigma2"]), "dB m²/m²", "Total continuum")
        add(sec, "k0h", f"{sp['k0h']:.4f}", "", "Perturbation-theory roughness parameter")
        add(sec, "Fa_atmospheric", f"{nz['FaA']:.2f}", "dB > kT0b", f"Median, {p.atmos_model}")
        add(sec, "Fa_man_made", f"{nz['FaM']:.2f}", "dB > kT0b")
        add(sec, "Fa_galactic", f"{nz['FaG']:.2f}" if math.isfinite(nz["FaG"]) else "screened",
            "dB > kT0b")
        add(sec, "Fa_total", f"{nz['Fa']:.2f}", "dB > kT0b", p.percentile)
        add(sec, "skywave_band_open", f"{sk['open']:.3f}", "0-1", "Interference weighting (f vs MUF)")
        add(sec, "vertical_reflection", f"{sk['vert_open']:.3f}", "0-1", "Self-clutter weighting (f vs foF2)")
        add(sec, "d_layer_absorption_vertical", f"{sk['abs_vert_db']:.2f}", "dB", "One-way")
    return rows


# =============================================================================
# 7. Input range checking
# =============================================================================
# Hard limits: (min, max, label, unit). Values outside these are rejected,
# because they are non-physical or outside the validity of a model component.
LIMITS = {
    "pt_w":                 (1.0, 10e6, "Transmit power", "W"),
    "duty":                 (0.001, 1.0, "Duty cycle", ""),
    "gain_dbi":             (-20.0, 30.0, "Tx/Rx gain", "dBi"),
    "cit_s":                (0.1, 3600.0, "Coherent integration time", "s"),
    "bandwidth_hz":         (1e3, 1e6, "Waveform bandwidth", "Hz"),
    "rx_beamwidth_deg":     (1.0, 360.0, "Rx azimuth beamwidth", "deg"),
    "look_az_deg":          (0.0, 360.0, "Boresight azimuth", "deg"),
    "losses_db":            (0.0, 40.0, "System/processing losses", "dB"),
    "rx_nf_db":             (0.0, 40.0, "Receiver noise figure", "dB"),
    "doppler_dr_db":        (-150.0, -20.0, "Doppler dynamic-range floor", "dB"),
    "pd_req":               (0.01, 0.999, "Required Pd", ""),
    "pfa":                  (1e-12, 0.1, "Pfa", ""),
    "rcs_m2":               (1e-4, 1e6, "RCS", "m²"),
    "target_h_m":           (0.0, 3000.0, "Target altitude", "m"),
    "target_vr_ms":         (-500.0, 500.0, "Target radial speed", "m/s"),
    "land_km":              (0.0, 20.0, "Land section length", "km"),
    "k_factor":             (0.5, 4.0, "Effective earth-radius factor", ""),
    "r_min_km":             (1.0, 100.0, "Minimum range", "km"),
    "r_max_km":             (20.0, 2000.0, "Maximum range computed", "km"),
    "r_step_km":            (0.1, 10.0, "Range step", "km"),
    "sea_state":            (0, 7, "Sea state", ""),
    "wind_from_deg":        (0.0, 360.0, "Wind direction", "deg"),
    "bragg_sd_hz":          (0.001, 1.0, "Bragg line broadening", "Hz"),
    "sky_interf_excess_db": (-30.0, 50.0, "Skywave interference excess", "dB"),
    "zenith_gain_db":       (-60.0, 10.0, "Gain towards ionosphere", "dB"),
    "iono_refl_loss_db":    (0.0, 40.0, "F-layer reflection loss", "dB"),
    "iono_doppler_sd_hz":   (0.01, 5.0, "Ionospheric Doppler spread", "Hz"),
    "iono_extent_km":       (0.0, 1000.0, "Self-clutter range extent", "km"),
    "speed_min_ms":         (-500.0, 500.0, "Speed sweep minimum", "m/s"),
    "speed_max_ms":         (-500.0, 500.0, "Speed sweep maximum", "m/s"),
    "speed_step_ms":        (0.01, 50.0, "Speed sweep step", "m/s"),
}
FREQ_LIMITS = (1.5, 30.0)          # P.368 / P.372 HF validity used here
TIME_LIMITS = {"month": (1, 12), "hour": (0, 23), "ssn": (0.0, 300.0)}
INTEGER_FIELDS = {"sea_state", "swerling", "month", "hour"}


def validate_inputs(p: RadarParams, month=None, hour=None, ssn=None, freq=None, site=None):
    """
    Range-check all inputs. Returns (errors, warnings), each a list of
    (field, message). Errors must be fixed before computing; warnings flag
    values that are legal but stretch a model's validity.
    """
    errors, warnings = [], []

    def chk(name, val, lo, hi, label, unit, integer=False):
        u = f" {unit}" if unit else ""
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            errors.append((name, f"{label}: must be a finite number"))
            return
        if integer and float(val) != int(val):
            errors.append((name, f"{label}: must be a whole number ({lo:g}-{hi:g})"))
            return
        if not (lo <= val <= hi):
            errors.append((name, f"{label} = {val:g}{u} is outside {lo:g} to {hi:g}{u}"))

    for name, (lo, hi, label, unit) in LIMITS.items():
        chk(name, getattr(p, name), lo, hi, label, unit, name in INTEGER_FIELDS)
    if p.swerling not in (0, 1):
        errors.append(("swerling", "Swerling model must be 0 or 1"))
    if p.sea not in GROUND_TYPES:
        errors.append(("sea", f"Unknown sea type '{p.sea}'"))
    if p.land not in GROUND_TYPES:
        errors.append(("land", f"Unknown land type '{p.land}'"))
    if p.atmos_model not in ATMOS_MODELS:
        errors.append(("atmos_model", f"Atmospheric noise must be one of: {', '.join(ATMOS_MODELS)}"))
    if p.man_made not in MAN_MADE:
        errors.append(("man_made", f"Unknown man-made noise category '{p.man_made}'"))

    # cross-field consistency
    if p.r_max_km < p.r_min_km + 10 * p.r_step_km:
        errors.append(("r_max_km", "Maximum range must exceed minimum range by at least 10 range steps"))
    if (p.r_max_km - p.r_min_km) / p.r_step_km > 20000:
        errors.append(("r_step_km", "Range grid too fine (more than 20 000 points) - increase the step"))
    if p.land_km >= p.r_min_km:
        errors.append(("land_km", f"Land section ({p.land_km:g} km) must be shorter than the minimum range ({p.r_min_km:g} km)"))
    if p.speed_max_ms <= p.speed_min_ms:
        errors.append(("speed_max_ms", "Speed sweep maximum must exceed the minimum"))
    elif (p.speed_max_ms - p.speed_min_ms) / p.speed_step_ms > 2000:
        errors.append(("speed_step_ms", "Speed sweep has more than 2000 points - increase the step"))

    # frequencies
    if not p.freqs_mhz:
        errors.append(("freqs", "At least one frequency is required"))
    elif len(p.freqs_mhz) > 60:
        errors.append(("freqs", "More than 60 frequencies - increase the step"))
    for f in list(p.freqs_mhz) + ([freq] if freq is not None else []):
        if not (FREQ_LIMITS[0] <= f <= FREQ_LIMITS[1]):
            errors.append(("freq" if f == freq else "freqs",
                           f"Frequency {f:g} MHz is outside {FREQ_LIMITS[0]:g}-{FREQ_LIMITS[1]:g} MHz"))
            break

    # time / solar
    for name, val in (("month", month), ("hour", hour), ("ssn", ssn)):
        if val is not None:
            lo, hi = TIME_LIMITS[name]
            chk(name, val, lo, hi, {"month": "Month", "hour": "UTC hour", "ssn": "Sunspot number"}[name],
                "", name in INTEGER_FIELDS)

    # --- warnings: legal but outside a model's comfortable validity -----------
    if errors:
        return errors, warnings
    if p.target_h_m > 1000:
        warnings.append(("target_h_m", "Altitude > 1 km: the target leaves the surface-wave regime; "
                                       "space-wave/line-of-sight effects are not modelled"))
    if ssn is not None and ssn > 160:
        warnings.append(("ssn", "SSN > 160: P.1239 foF2 is saturated at 160, so values above add nothing"))
    fmax = max(list(p.freqs_mhz) + ([freq] if freq else []))
    if fmax > 20 and p.percentile.startswith("Upper"):
        warnings.append(("freqs", "Above 20 MHz the P.372 atmospheric-noise deciles are held at their 20 MHz values"))
    if p.sea_clutter:
        k0 = 2 * math.pi * fmax * 1e6 / C0
        k0h = k0 * SEA_STATES[int(p.sea_state)][1] / 4
        if k0h > 0.6:
            warnings.append(("sea_state", f"k0·h = {k0h:.2f} at {fmax:g} MHz: beyond the small-perturbation "
                                          "regime; second-order clutter is capped and indicative only"))
    if p.atmos_model == ATMOS_MODELS[1]:
        pa = PARAMETRIC_ATMOS
        fmin = min(list(p.freqs_mhz) + ([freq] if freq else []))
        if fmin < pa["f_min"] or fmax > pa["f_max"]:
            warnings.append(("atmos_model", f"Parametric atmospheric noise is fitted for {pa['f_min']:g}-"
                                            f"{pa['f_max']:g} MHz; outside that it is extrapolated"))
        if site is not None and (abs(site[0] - pa["lat"]) > 5 or abs(site[1] - pa["lon"]) > 10):
            warnings.append(("atmos_model", "Parametric atmospheric noise was fitted for a different site - "
                                            "refit with fit_parametric_atmos() or use the P.372 tables"))
    if p.cit_s > 300:
        warnings.append(("cit_s", "Integration > 300 s: assumes the target stays in one range-Doppler cell"))
    if p.duty < 1 and p.duty * p.cit_s < 0.1:
        warnings.append(("duty", "Very low energy on target (duty × CIT < 0.1 s)"))
    if p.rx_beamwidth_deg > 180:
        warnings.append(("rx_beamwidth_deg", "Beamwidth > 180°: clutter cell includes the landward sector"))
    if not p.sea_clutter and abs(p.target_vr_ms) < 15:
        warnings.append(("sea_clutter", "Slow target with sea clutter disabled: results will be optimistic"))
    return errors, warnings


class HFSWRModel:
    def __init__(self, site_json):
        self.noise = SiteNoise(site_json)
        self._gw_cache = {}

    # -- propagation (cached; independent of time) ----------------------------
    def path_gain_db(self, p: RadarParams, f, h=None):
        h = p.target_h_m if h is None else h
        key = (f, p.sea, p.land, p.land_km, h, p.k_factor, p.r_min_km, p.r_max_km, p.r_step_km)
        if key not in self._gw_cache:
            r = np.arange(p.r_min_km, p.r_max_km + 1e-9, p.r_step_km) * 1e3
            wdb = mixed_path_W_db(f, r, GROUND_TYPES[p.land], GROUND_TYPES[p.sea],
                                  p.land_km * 1e3, h, p.k_factor)
            self._gw_cache[key] = (r, wdb)
        return self._gw_cache[key]

    def prepare(self, p, f):
        self.path_gain_db(p, f)
        if p.sea_clutter:
            self.path_gain_db(p, f, h=0.0)

    # -- full SINR profile -------------------------------------------------------
    def sinr_profile(self, p: RadarParams, f, month, hour, ssn, components=False):
        """SINR (dB) after coherent integration versus range."""
        lam = C0 / (f * 1e6)
        r, wdb = self.path_gain_db(p, f)
        common = (10 * np.log10(p.pt_w * p.duty) + 2 * p.gain_dbi + 20 * np.log10(lam)
                  - 30 * np.log10(4 * np.pi) - 40 * np.log10(r) - p.losses_db)
        # target energy (J)
        s_db = common + 10 * np.log10(p.cit_s) + 10 * np.log10(p.rcs_m2) + 2 * wdb
        S = 10 ** (s_db / 10)

        # noise + skywave interference density (W/Hz)
        nz = self.noise.total(month, hour, ssn, f, p.man_made, p.percentile,
                              p.galactic_screening, p.atmos_model)
        fa = 10 ** (nz["Fa"] / 10)
        fi = 0.0
        sky = self.skywave_state(month, hour, ssn, f)
        if p.sky_interference:
            med = self.noise.total(month, hour, ssn, f, p.man_made, "Median",
                                   p.galactic_screening, p.atmos_model)["Fa"]
            fi = sky["open"] * 10 ** ((med + p.sky_interf_excess_db - sky["abs_oblique_db"]) / 10)
        N0 = K_BOLTZ * T0 * (fa + fi + 10 ** (p.rx_nf_db / 10) - 1.0)
        N = np.full_like(r, N0)

        fd = 2.0 * p.target_vr_ms / lam
        bin_hz = 1.0 / p.cit_s
        dr_lin = 10 ** (p.doppler_dr_db / 10)
        cell_m = C0 / (2.0 * p.bandwidth_hz)

        # sea clutter density (W/Hz)
        Csea = np.zeros_like(r)
        if p.sea_clutter:
            _, w0 = self.path_gain_db(p, f, h=0.0)
            spec = sea_clutter_spectrum(f, p.sea_state, p.wind_from_deg, p.look_az_deg,
                                        p.rx_beamwidth_deg)
            area = r * math.radians(p.rx_beamwidth_deg) * cell_m
            pc_db = common + 2 * w0 + 10 * np.log10(area)          # W per unit sigma0
            Pc = 10 ** (pc_db / 10)
            sd = math.sqrt(p.bragg_sd_hz ** 2 + (bin_hz / math.sqrt(12)) ** 2)
            p1, p2 = sea_clutter_psd(fd, spec, sd)
            s1tot = spec["sigma1_pos"] + spec["sigma1_neg"]
            Csea = Pc * (p1 + p2 + dr_lin * p.cit_s * (s1tot + spec["sigma2"]))

        # ionospheric self-clutter density (W/Hz)
        Cion = np.zeros_like(r)
        if p.iono_clutter and sky["vert_open"] > 1e-4:
            hp = sky["hF_km"] * 1e3
            pi_db = (10 * np.log10(p.pt_w * p.duty) + 2 * (p.gain_dbi + p.zenith_gain_db)
                     + 20 * np.log10(lam) - 20 * np.log10(4 * np.pi) - 20 * np.log10(2 * hp)
                     - p.iono_refl_loss_db - 2 * sky["abs_vert_db"] - p.losses_db)
            Pi = 10 ** (pi_db / 10) * sky["vert_open"]
            ext = p.iono_extent_km * 1e3
            frac = np.where((r >= hp) & (r <= hp + ext), min(cell_m / ext, 1.0), 0.0)
            sdi = math.sqrt(p.iono_doppler_sd_hz ** 2 + (bin_hz / math.sqrt(12)) ** 2)
            psd = math.exp(-0.5 * (fd / sdi) ** 2) / (sdi * math.sqrt(2 * math.pi))
            Cion = Pi * frac * (psd + dr_lin * p.cit_s)

        sinr = 10 * np.log10(S / (N + Csea + Cion))
        if not components:
            return r, sinr
        return r, sinr, {"S": S, "N": N, "Csea": Csea, "Cion": Cion, "sky": sky,
                         "noise": nz, "fd": fd}

    def skywave_state(self, month, hour, ssn, f):
        fo = self.noise.fof2(month, hour, ssn)
        m3 = self.noise.m3000(month, hour, ssn)
        muf = fo * m3
        chi = solar_zenith_deg(self.noise.lat, self.noise.lon, month, hour)
        a_v = d_layer_absorption_db(f, chi, ssn)
        logistic = lambda x: 1.0 / (1.0 + math.exp(-max(min(x, 50), -50)))
        return {"foF2": fo, "M3000": m3, "MUF3000": muf, "chi": chi,
                "hF_km": 1490.0 / m3 - 176.0,              # Shimazaki hmF2
                "open": logistic((muf - f) / (0.05 * muf)),
                "vert_open": logistic((fo - f) / (0.03 * fo)),
                "abs_vert_db": a_v, "abs_oblique_db": 3.0 * a_v}

    # -- detection --------------------------------------------------------------
    def pd_profile(self, p, f, month, hour, ssn):
        r, sinr = self.sinr_profile(p, f, month, hour, ssn)
        return r, pd_from_snr(10 ** (sinr / 10), p.pfa, p.swerling)

    def detection_range_km(self, p, f, month, hour, ssn, snr_req=None):
        snr_req = p.snr_req_db if snr_req is None else snr_req
        r, sinr = self.sinr_profile(p, f, month, hour, ssn)
        below = np.nonzero(sinr < snr_req)[0]
        if len(below) == 0:
            return r[-1] / 1e3
        i = below[0]
        if i == 0:
            return 0.0
        r0, r1, s0, s1 = r[i - 1], r[i], sinr[i - 1], sinr[i]
        return (r0 + (r1 - r0) * (s0 - snr_req) / (s0 - s1)) / 1e3

    # -- sweeps -----------------------------------------------------------------
    SWEEPS = {"diurnal": np.arange(0, 25), "annual": np.arange(1, 13),
              "sunspot": np.arange(0, 201, 10)}

    def sweep_values(self, p, kind):
        if kind == "speed":
            n = int(round((p.speed_max_ms - p.speed_min_ms) / p.speed_step_ms))
            return p.speed_min_ms + p.speed_step_ms * np.arange(n + 1)
        return self.SWEEPS[kind]

    def _args(self, p, kind, x, month, hour, ssn):
        """(params, month, hour, ssn) for one point of a sweep."""
        if kind == "diurnal":
            return p, month, int(x) % 24, ssn
        if kind == "annual":
            return p, int(x), hour, ssn
        if kind == "speed":
            return replace(p, target_vr_ms=float(x)), month, hour, ssn
        return p, month, hour, float(x)

    def range_sweep(self, p, kind, month, hour, ssn):
        xs = self.sweep_values(p, kind)
        req = p.snr_req_db
        out = {}
        for f in p.freqs_mhz:
            vals = []
            for x in xs:
                pp, mo, hr, ss = self._args(p, kind, x, month, hour, ssn)
                vals.append(self.detection_range_km(pp, f, mo, hr, ss, req))
            out[f] = np.array(vals)
        return xs, out

    def pd_sweep(self, p, kind, f, month, hour, ssn):
        """Pd matrix [len(x), len(range)] at a single frequency."""
        xs = self.sweep_values(p, kind)
        rows = []
        r = None
        for x in xs:
            r, pd = self.pd_profile(*self._args(p, kind, x, month, hour, ssn)[:1], f,
                                    *self._args(p, kind, x, month, hour, ssn)[1:])
            rows.append(pd)
        return xs, r / 1e3, np.array(rows)


def default_site_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "site_lands_end.json")
