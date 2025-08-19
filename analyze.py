
import argparse
import numpy as np
import pandas as pd
from eda_core import hp_phasic, impulse_response, fista_nonneg_l1_deconv, central_state_from_driver, compute_features, conv_full_trim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='Input CSV with columns time,signal[,channel]')
    ap.add_argument('--fs', type=float, required=True, help='Sampling rate in Hz')
    ap.add_argument('--mode', choices=['SC','SP'], default='SC', help='Signal mode (only affects interpretation)')
    ap.add_argument('--tau1', type=float, default=0.7, help='Impulse response fast tau [s]')
    ap.add_argument('--tau2', type=float, default=2.0, help='Impulse response slow tau [s]')
    ap.add_argument('--lam', type=float, default=0.03, help='L1 sparsity weight')
    ap.add_argument('--baseline_win', type=float, default=10.0, help='Baseline moving-average window [s]')
    ap.add_argument('--stim', type=str, default='', help='Comma-separated stimulus times in seconds (optional)')
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    fs = float(args.fs)
    stimuli = None
    if args.stim:
        stimuli = np.array([float(x) for x in args.stim.split(',') if x.strip()!=''], dtype=float)

    phasic_raw, tonic = hp_phasic(df['signal'].to_numpy().astype(float), fs, window_sec=args.baseline_win)
    h = impulse_response(args.tau1, args.tau2, fs=fs, dur_s=10.0)
    driver = fista_nonneg_l1_deconv(phasic_raw, h, lam=args.lam, n_iter=600)
    phasic_fit = conv_full_trim(driver, h, n_out=len(df))

    feats = compute_features(df['signal'].to_numpy(), phasic_fit, tonic, driver, fs=fs, stimuli_s=stimuli)

    pd.DataFrame({"time": df["time"], "driver": driver}).to_csv("driver_estimate.csv", index=False)
    pd.DataFrame({"time": df["time"], "phasic_fit": phasic_fit, "tonic": tonic}).to_csv("phasic_fit.csv", index=False)
    pd.DataFrame([feats.__dict__]).to_csv("features.csv", index=False)
    print("Saved: driver_estimate.csv, phasic_fit.csv, features.csv")

if __name__ == "__main__":
    main()
