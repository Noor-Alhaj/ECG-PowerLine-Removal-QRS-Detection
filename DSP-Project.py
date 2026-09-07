# Noor Alhaj 1221543
# Dina Daoud 1221928
# Areej Younis 1221419
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import matplotlib
matplotlib.use('TkAgg')

plt.ion()

# -------------------------------------------------
# Pan-Tompkins Algorithm
# -------------------------------------------------
def pan_tompkins(ecg, fs=360):
    nyq = fs / 2
    b, a = butter(4, [5/nyq, 15/nyq], btype='band')
    filtered = filtfilt(b, a, ecg)

    deriv = np.diff(filtered, append=filtered[-1])
    squared = deriv ** 2

    win = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(win)/win, mode='same')

    thresh = np.mean(integrated) + 0.6 * np.std(integrated)
    min_dist = int(0.25 * fs)
    candidates, _ = find_peaks(integrated, height=thresh, distance=min_dist)

    search = int(0.05 * fs)
    peaks = []
    for c in candidates:
        s = max(0, c-search)
        e = min(len(ecg), c+search)
        peak = s + np.argmax(np.abs(filtered[s:e]))
        peaks.append(peak)

    return np.array(peaks, dtype=int)

# -------------------------------------------------
# Metrics
# -------------------------------------------------
def residual_50hz(signal, fs):
    f = np.fft.rfftfreq(len(signal), 1/fs)
    mag = np.abs(np.fft.rfft(signal))
    idx = np.where((f >= 49) & (f <= 51))[0]
    return np.max(mag[idx]) if len(idx) > 0 else 0  

def compute_snr(clean, filtered):
    noise = filtered - clean
    return 10 * np.log10(np.sum(clean**2) / np.sum(noise**2))

def qrs_metrics(detected, reference, fs=360, tol=0.1):
    tol = int(tol * fs)
    TP = 0
    matched_refs = set()  
    for d in detected:
        for i, r in enumerate(reference):
            if i not in matched_refs and abs(d - r) <= tol:
                TP += 1
                matched_refs.add(i)
                break
    FN = len(reference) - TP
    FP = len(detected) - TP
    return TP, FN, FP

# -------------------------------------------------
# Load ECG (LOCAL FILES)
# -------------------------------------------------
print("Loading Record 100 locally...")
record = wfdb.rdrecord('100')
ann = wfdb.rdann('100', 'atr')

ecg_raw = record.p_signal[:, 0]
fs = int(record.fs)
true_peaks = ann.sample
t = np.arange(len(ecg_raw)) / fs

print("fs =", fs, "Hz")
print("Samples =", len(ecg_raw))
print("Annotations =", len(true_peaks))

# -------------------------------------------------
# Add Power-Line Interference
# -------------------------------------------------
pli_amp = 0.2 * np.ptp(ecg_raw)

# Fixed 50 Hz
pli_fixed = pli_amp * np.sin(2*np.pi*50*t)
ecg_noisy_fixed = ecg_raw + pli_fixed

# Drifting 49.5–50.5 Hz
drift = 0.5 * np.sin(2*np.pi*t/(len(t)/fs))
f_drift = 50 + drift
pli_drift = pli_amp * np.sin(2*np.pi*np.cumsum(f_drift)/fs)
ecg_noisy_drift = ecg_raw + pli_drift

# -------------------------------------------------
# Time Window (10 seconds)
# -------------------------------------------------
duration = 10
N = int(duration * fs)

# -------------------------------------------------
# Raw ECG
# -------------------------------------------------
plt.figure(figsize=(12,4))
plt.plot(t[:N], ecg_noisy_fixed[:N])
plt.title("Raw ECG with 50 Hz Power-Line Interference")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show(block=False)

# -------------------------------------------------
# FFT of Noisy ECG
# -------------------------------------------------
f = np.fft.rfftfreq(len(ecg_noisy_fixed), 1/fs)
fft_noisy = np.abs(np.fft.rfft(ecg_noisy_fixed))

plt.figure(figsize=(12,4))
plt.plot(f, fft_noisy)
plt.xlim(0, 100)
plt.title("FFT of Noisy ECG (50 Hz Interference)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show(block=False)

# -------------------------------------------------
# Fixed 50 Hz Notch Filter on Fixed Noise
# -------------------------------------------------
Q = 30
b, a = iirnotch(50/(fs/2), Q)
ecg_fixed = filtfilt(b, a, ecg_noisy_fixed)

plt.figure(figsize=(12,4))
plt.plot(t[:N], ecg_fixed[:N])
plt.title("ECG After Fixed 50 Hz Notch Filter")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show(block=False)

fft_fixed = np.abs(np.fft.rfft(ecg_fixed))
plt.figure(figsize=(12,4))
plt.plot(f, fft_fixed)
plt.xlim(0, 100)
plt.title("FFT After Fixed 50 Hz Notch Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show(block=False)

# -------------------------------------------------
# Fixed 50 Hz Notch Filter on Drifting Noise
# -------------------------------------------------
ecg_fixed_on_drift = filtfilt(b, a, ecg_noisy_drift)

# -------------------------------------------------
# Adaptive FFT-Based Notch Filter
# -------------------------------------------------
segment_sec = 5
segN = int(segment_sec * fs)
step = segN // 2

ecg_adaptive = np.zeros_like(ecg_noisy_drift)
weight = np.zeros_like(ecg_noisy_drift)

freq_estimates = []
time_estimates = []

start = 0
while start + segN <= len(ecg_noisy_drift):
    seg = ecg_noisy_drift[start:start+segN]

    f_seg = np.fft.rfftfreq(segN, 1/fs)
    mag_seg = np.abs(np.fft.rfft(seg))
    idx = np.where((f_seg >= 49) & (f_seg <= 51))[0]
    est_freq = f_seg[idx[np.argmax(mag_seg[idx])]]

    freq_estimates.append(est_freq)
    time_estimates.append((start + segN//2) / fs)

    b_ad, a_ad = iirnotch(est_freq/(fs/2), Q)
    seg_filt = filtfilt(b_ad, a_ad, seg)

    win = np.hanning(segN)
    ecg_adaptive[start:start+segN] += seg_filt * win
    weight[start:start+segN] += win

    start += step

# -----------------------------------------------
# Strongest estimated interference frequency near 50 Hz
# -----------------------------------------------
freq_estimates = np.array(freq_estimates)

# strongest peak = most frequent / dominant estimate
strongest_freq = np.mean(freq_estimates)

print(f"Strongest estimated power-line frequency near 50 Hz = {strongest_freq:.2f} Hz")

nz = weight > 0
ecg_adaptive[nz] /= weight[nz]
ecg_adaptive[~nz] = ecg_noisy_drift[~nz]

plt.figure(figsize=(12,4))
plt.plot(t[:N], ecg_adaptive[:N])
plt.title("ECG After Adaptive FFT-Based Notch Filter")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show(block=False)

fft_adapt = np.abs(np.fft.rfft(ecg_adaptive))
plt.figure(figsize=(12,4))
plt.plot(f, fft_adapt)
plt.xlim(0, 100)
plt.title("FFT After Adaptive Notch Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show(block=False)

# -------------------------------------------------
# Frequency Tracking Plot 
# -------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(time_estimates, freq_estimates, 'o-', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Estimated Frequency (Hz)")
plt.title("Estimated Power-Line Frequency Tracking")
plt.grid()
plt.show(block=False)

# -------------------------------------------------
# Pan-Tompkins QRS Detection
# -------------------------------------------------
peaks_noisy_fixed = pan_tompkins(ecg_noisy_fixed, fs)
peaks_fixed = pan_tompkins(ecg_fixed, fs)
peaks_noisy_drift = pan_tompkins(ecg_noisy_drift, fs) 
peaks_fixed_on_drift = pan_tompkins(ecg_fixed_on_drift, fs)  
peaks_adapt = pan_tompkins(ecg_adaptive, fs)

def plot_qrs(signal, peaks, title):
    p = peaks[peaks < N]
    plt.figure(figsize=(12,4))
    plt.plot(t[:N], signal[:N])
    plt.plot(t[p], signal[p], 'ro')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show(block=False)

plot_qrs(ecg_noisy_fixed, peaks_noisy_fixed, "Pan-Tompkins on Noisy ECG (Fixed)")
plot_qrs(ecg_fixed, peaks_fixed, "Pan-Tompkins After Fixed Notch (Fixed Noise)")
plot_qrs(ecg_noisy_drift, peaks_noisy_drift, "Pan-Tompkins on Noisy ECG (Drifting)")  
plot_qrs(ecg_fixed_on_drift, peaks_fixed_on_drift, "Pan-Tompkins After Fixed Notch (Drifting Noise)") 
plot_qrs(ecg_adaptive, peaks_adapt, "Pan-Tompkins After Adaptive Notch")

# -------------------------------------------------
# Performance Evaluation
# -------------------------------------------------
res_noisy_drift = residual_50hz(ecg_noisy_drift, fs)  
res_fixed_on_drift = residual_50hz(ecg_fixed_on_drift, fs)  
res_adapt = residual_50hz(ecg_adaptive, fs)

snr_fixed_on_drift = compute_snr(ecg_raw, ecg_fixed_on_drift) 
snr_adapt = compute_snr(ecg_raw, ecg_adaptive)

tp_nd, fn_nd, fp_nd = qrs_metrics(peaks_noisy_drift, true_peaks, fs)  
tp_fd, fn_fd, fp_fd = qrs_metrics(peaks_fixed_on_drift, true_peaks, fs)  
tp_a, fn_a, fp_a = qrs_metrics(peaks_adapt, true_peaks, fs)

print("\n" + "="*90)
print("Comparison Results for Drifting Noise".center(90))
print("="*90)

header = (
    f"{'Metric':<25}"
    f"{'Noisy Drifting':>20}"
    f"{'Fixed on Drifting':>22}"
    f"{'Adaptive':>15}"
)
print(header)
print("-"*90)

print(f"{'Residual 50 Hz':<25}{res_noisy_drift:>20.2f}{res_fixed_on_drift:>22.2f}{res_adapt:>15.2f}")
print(f"{'SNR (dB)':<25}{'N/A':>20}{snr_fixed_on_drift:>22.2f}{snr_adapt:>15.2f}")

print("-"*90)

print(f"{'TP (True Positives)':<25}{tp_nd:>20}{tp_fd:>22}{tp_a:>15}")
print(f"{'FN (Missed Beats)':<25}{fn_nd:>20}{fn_fd:>22}{fn_a:>15}")
print(f"{'FP (False Detections)':<25}{fp_nd:>20}{fp_fd:>22}{fp_a:>15}")

print("-"*90)

acc_nd = tp_nd / (tp_nd + fn_nd + fp_nd) * 100
acc_fd = tp_fd / (tp_fd + fn_fd + fp_fd) * 100
acc_a  = tp_a  / (tp_a  + fn_a  + fp_a ) * 100

print(f"{'QRS Accuracy (%)':<25}{acc_nd:>20.2f}{acc_fd:>22.2f}{acc_a:>15.2f}")
print("="*90)

plt.show(block=True)