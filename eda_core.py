
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win < 1:
        return x.copy()
    kernel = np.ones(win) / win
    pad = win // 2
    x_pad = np.pad(x, (pad, pad - (win % 2 == 0)), mode='edge')
    y = np.convolve(x_pad, kernel, mode='valid')
    return y

def hp_phasic(x: np.ndarray, fs: float, window_sec: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    win = max(1, int(window_sec * fs))
    tonic = moving_average(x, win)
    phasic = x - tonic
    return phasic, tonic

def impulse_response(tau1: float, tau2: float, fs: float, dur_s: float = 10.0) -> np.ndarray:
    n = int(dur_s * fs)
    t = np.arange(n) / fs
    if abs(tau2 - tau1) < 1e-6:
        tau2 = tau1 + 1e-6
    h = (np.exp(-t / tau2) - np.exp(-t / tau1)) / (tau2 - tau1)
    h = h / (h.max() + 1e-12)
    return h

def conv_full_trim(a: np.ndarray, b: np.ndarray, n_out: int) -> np.ndarray:
    z = np.convolve(a, b, mode='full')
    return z[:n_out]

def convT_full_trim(resid: np.ndarray, h: np.ndarray, n_out: int) -> np.ndarray:
    h_rev = h[::-1]
    z = np.convolve(resid, h_rev, mode='full')
    return z[:n_out]

def fista_nonneg_l1_deconv(y: np.ndarray, h: np.ndarray, lam: float = 0.01, n_iter: int = 500, step: float = None) -> np.ndarray:
    n = len(y)
    s = np.zeros(n)
    z = s.copy()
    t = 1.0
    if step is None:
        L = 2.0 * np.sum(h * h)
        step = 1.0 / (L + 1e-9)

    def grad(z_vec: np.ndarray) -> np.ndarray:
        Hz = conv_full_trim(z_vec, h, n_out=n)
        r = Hz - y
        g = convT_full_trim(r, h, n_out=n)
        return g

    for k in range(n_iter):
        g = grad(z)
        s_next = z - step * g
        s_next = np.maximum(0.0, np.abs(s_next) - lam * step) * np.sign(s_next)
        s_next = np.maximum(0.0, s_next)
        t_next = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
        z = s_next + ((t - 1) / t_next) * (s_next - s)
        s, t = s_next, t_next
    return s

def central_state_from_driver(s: np.ndarray, fs: float, tau_c: float = 5.0, kappa: float = 1.0) -> np.ndarray:
    dt = 1.0 / fs
    a = max(0.0, min(1.0, 1.0 - dt / max(1e-6, tau_c)))
    z = np.zeros_like(s)
    for i in range(1, len(s)):
        z[i] = a * z[i-1] + kappa * s[i]
    return z

@dataclass
class EDAFeatures:
    mean_phasic_power: float
    burst_rate_hz: float
    mean_burst_amp: float
    median_burst_amp: float
    max_burst_amp: float
    nonresponse_ratio: float
    habituation_slope: float
    mean_recovery_time_s: float
    tonic_level_mean: float
    tonic_level_std: float

def detect_bursts(s: np.ndarray, fs: float, thr: float = None, refractory_s: float = 0.5):
    if thr is None:
        mu = np.median(s)
        mad = np.median(np.abs(s - mu)) + 1e-9
        z = (s - mu) / (1.4826 * mad)
        thr = 3.0
    indices = []
    amps = []
    last_idx = -int(refractory_s * fs)
    for i in range(1, len(s) - 1):
        if (i - last_idx) < int(refractory_s * fs):
            continue
        if s[i] > s[i-1] and s[i] >= s[i+1]:
            mu = np.median(s[max(0, i-500):i+1])
            mad = np.median(np.abs(s[max(0, i-500):i+1] - mu)) + 1e-9
            z = (s[i] - mu) / (1.4826 * mad)
            if z >= thr and s[i] > 0:
                indices.append(i)
                amps.append(float(s[i]))
                last_idx = i
    return np.array(indices, dtype=int), np.array(amps, dtype=float)

def recovery_time(phasic: np.ndarray, peak_idx: int, fs: float, frac: float = 0.5, max_s: float = 10.0) -> float:
    peak_val = phasic[peak_idx]
    target = peak_val * frac
    nmax = min(len(phasic), peak_idx + int(max_s * fs))
    for j in range(peak_idx, nmax):
        if phasic[j] <= target:
            return (j - peak_idx) / fs
    return float('nan')

def compute_features(signal: np.ndarray, phasic: np.ndarray, tonic: np.ndarray, driver: np.ndarray, fs: float,
                     stimuli_s: np.ndarray = None) -> EDAFeatures:
    idx, amps = detect_bursts(driver, fs=fs)
    burst_rate = len(idx) / (len(driver) / fs)

    nonresp = np.nan
    hab_slope = np.nan
    rec_times = []

    if stimuli_s is not None and len(stimuli_s) > 0:
        window = 3.0
        hits = 0
        seq_amps = []
        for n, t0 in enumerate(stimuli_s):
            i0 = int(t0 * fs)
            i1 = min(len(driver) - 1, int((t0 + window) * fs))
            j = np.where((idx >= i0) & (idx <= i1))[0]
            if len(j) > 0:
                hits += 1
                seq_amps.append(float(np.max(amps[j])))
        nonresp = 1.0 - (hits / len(stimuli_s))
        if len(seq_amps) >= 2:
            x = np.arange(len(seq_amps))
            y = np.array(seq_amps)
            xs = (x - x.mean())
            denom = (xs**2).sum() + 1e-9
            hab_slope = float((xs * (y - y.mean())).sum() / denom)

    if len(idx) > 0:
        peak_candidates = idx[:min(10, len(idx))]
        for pk in peak_candidates:
            rt = recovery_time(phasic, pk, fs=fs, frac=0.5)
            if not np.isnan(rt):
                rec_times.append(rt)

    feats = EDAFeatures(
        mean_phasic_power=float(np.mean(phasic * phasic)),
        burst_rate_hz=float(burst_rate),
        mean_burst_amp=float(np.mean(amps)) if len(amps) else float('nan'),
        median_burst_amp=float(np.median(amps)) if len(amps) else float('nan'),
        max_burst_amp=float(np.max(amps)) if len(amps) else float('nan'),
        nonresponse_ratio=float(nonresp) if not np.isnan(nonresp) else float('nan'),
        habituation_slope=float(hab_slope) if not np.isnan(hab_slope) else float('nan'),
        mean_recovery_time_s=float(np.nanmean(rec_times)) if len(rec_times) else float('nan'),
        tonic_level_mean=float(np.mean(tonic)),
        tonic_level_std=float(np.std(tonic)),
    )
    return feats
