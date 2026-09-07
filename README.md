# ECG Power-Line Removal & Pan-Tompkins QRS Detection

Removing 50 Hz power-line interference from an ECG signal using two filters — a **fixed notch** and an **FFT-based adaptive notch** — then detecting R-peaks with the Pan-Tompkins algorithm and comparing the results.

Real power grids drift by a few tenths of a hertz around 50 Hz. A notch filter fixed at exactly 50 Hz cuts a narrow slot and misses the interference once it moves; the adaptive filter re-measures the interference frequency every 5 seconds and moves the notch to follow it.


## Running

```bash
pip install wfdb numpy scipy matplotlib
python DSP-Project.py
```

Run from the directory containing `100.dat`, `100.hea`, and `100.atr` — the record is opened by relative name. Plots open as the script runs and the comparison table is printed to the terminal.

`DSP-phase1.py` is the earlier phase 1 script; it downloads the record from PhysioNet instead of reading local files.

## Data

MIT-BIH Arrhythmia Database record 100: 30 minutes of ECG at 360 Hz, lead MLII, with 2 274 annotated beats used as ground truth. The record is clean, so the 50 Hz interference is added synthetically at 20 % of the signal's peak-to-peak amplitude — either at a fixed frequency or drifting across 49.5–50.5 Hz.

## Results

With drifting interference, over the full record:

| Metric | Noisy | Fixed notch | Adaptive notch |
| --- | ---: | ---: | ---: |
| Residual 50 Hz magnitude | 18 922 | 4 961 | **348** |
| SNR vs. clean signal | −4.19 dB | 11.16 dB | **29.12 dB** |
| QRS accuracy | 99.96 % | 99.96 % | 99.96 % |

The adaptive filter removes the interference far better — 14× less residual and 18 dB more SNR — because the fixed notch only attenuates energy sitting exactly at 50 Hz, while drifting interference spends most of its time elsewhere.

QRS detection is unaffected either way. Pan-Tompkins starts with a 5–15 Hz bandpass that already rejects 50 Hz, so all three signals give the same 2 273 true positives, 1 missed beat, and 0 false detections. The filtering matters for signal quality and waveform morphology, not for this detector.

## How it works

- **Fixed notch** — `scipy.signal.iirnotch` at 50 Hz with Q = 30, applied via `filtfilt` for zero-phase filtering so R-peak timing is not shifted.
- **Adaptive notch** — the signal is split into 5-second segments overlapping by 50 %. Each segment's FFT gives the strongest bin between 49 and 51 Hz, a notch is designed at that frequency, and the filtered segments are recombined by Hann-weighted overlap-add.
- **Pan-Tompkins** — bandpass 5–15 Hz, differentiate, square, integrate over 150 ms, then pick peaks above `mean + 0.6·std` with a 250 ms refractory period and refine each one to the true R-peak.
- **Scoring** — residual 50 Hz is the peak FFT magnitude in the 49–51 Hz band; SNR is measured against the clean signal; detections are matched to annotations within 100 ms.

## Files

| File | Description |
| --- | --- |
| `DSP-Project.py` | Final implementation: both filters, QRS detection, comparison table. |
| `DSP-phase1.py` | Phase 1: loading, FFT, fixed notch, initial detection. |
| `100.dat`, `100.hea`, `100.atr` | MIT-BIH record 100. |
| `dsp_ieepaper.pdf` | Final report (IEEE format) with all plots. |
