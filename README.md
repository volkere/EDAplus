
# EDA Parkinson Demo Toolkit
(for howto setup and measure skin potential reaction (SPR) please refer to https://github.com/volkere/gsr_sensor)

This package contains a minimal, dependency-light pipeline to analyze electrodermal signals (SC or SP):
- preprocessing (tonic/phasic separation via moving-average)
- deconvolution (FISTA, nonnegative L1) to estimate sudomotor driver
- simple central arousal proxy (leaky integrator over driver)
- feature extraction (burst rate, amplitudes, non-response ratio, habituation slope, recovery time, tonic stats)

## Files
- `demo_signal.csv`: synthetic 300 s SC recording @ 10 Hz
- `analyze.py`: run analysis on a CSV (time,signal,channel)
- `eda_core.py`: reusable functions (impulse response, FISTA deconvolution, features)
- `features.csv`: features for the demo
- `driver_estimate.csv`: estimated driver time series
- `phasic_fit.csv`: phasic fit and tonic

## Usage

```bash
python analyze.py --in demo_signal.csv --fs 10 --mode SC
# Or with your own CSV:
# python analyze.py --in /path/to/your.csv --fs 32 --mode SP --tau1 0.7 --tau2 2.0 --lam 0.03
```

Expected CSV columns:
- `time` (seconds)
- `signal` (EDA raw channel)
- `channel` (optional: "SC" or "SP")

Outputs:
- `features.csv` with summary metrics
- `driver_estimate.csv` with the sudomotor driver
- `phasic_fit.csv` with reconstructed phasic and tonic
