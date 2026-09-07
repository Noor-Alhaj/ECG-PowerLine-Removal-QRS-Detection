# Noor Alhaj-1221543
# Dina Daoud-1221928
# Areej Younis-1221419

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

import matplotlib
matplotlib.use('TkAgg')  

plt.ion()

def pan_tompkins(ecg, fs=360):
    """Accurate QRS detection with refinement to place marker exactly on R-peak"""
    nyq = fs / 2
    b_bp, a_bp = butter(4, [5/nyq, 15/nyq], btype='band')
    filtered = filtfilt(b_bp, a_bp, ecg)
    
    deriv = np.diff(filtered)
    deriv = np.append(deriv, 0)
    squared = deriv ** 2
    
    win = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(win)/win, mode='same')
    
    thresh = np.mean(integrated) + 0.6 * np.std(integrated)
    min_dist = int(0.25 * fs)
    candidates, _ = find_peaks(integrated, height=thresh, distance=min_dist)
    
    search = int(0.05 * fs)  
    peaks = []
    for c in candidates:
        s = max(0, c - search)
        e = min(len(ecg), c + search)
        peak_idx = s + np.argmax(np.abs(filtered[s:e]))
        peaks.append(peak_idx)
    
    return np.array(peaks)

print("Loading MIT-BIH Record 100...")
record = wfdb.rdrecord('100', pn_dir='mitdb')
ecg_raw = record.p_signal[:, 0] 
fs = record.fs
print(f"Sampling frequency: {fs} Hz, Length: {len(ecg_raw)} samples")

# Synthetic interference
t = np.arange(len(ecg_raw)) / fs
pli_amplitude = 0.2 * np.ptp(ecg_raw)

# Fixed 50 Hz for Tasks 1-2
pli_fixed = pli_amplitude * np.sin(2 * np.pi * 50 * t)
ecg_noisy_fixed = ecg_raw + pli_fixed

# Drifting (±0.2 Hz) for Task 3
drift = 0.2 * np.sin(2 * np.pi * t / (len(ecg_raw)/fs))
f_drift = 50 + drift
pli_drift = pli_amplitude * np.sin(2 * np.pi * np.cumsum(f_drift) / fs)
ecg_noisy_drift = ecg_raw + pli_drift

duration = 10
samples_10s = int(duration * fs)
time_10s = t[:samples_10s]

# Dedicated 10-second plot
plt.figure(figsize=(12, 5))
plt.plot(time_10s, ecg_noisy_fixed[:samples_10s], 'b')
plt.title('Task 1: Raw ECG with Synthetic 50 Hz Interference (First 10 seconds)')
plt.xlabel('Time (s)'); plt.ylabel('Amplitude (mV)')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# FFT noisy (no red line)
f = np.fft.rfftfreq(len(ecg_noisy_fixed), 1/fs)
fft_noisy = np.abs(np.fft.rfft(ecg_noisy_fixed))
plt.figure(figsize=(12, 5))
plt.plot(f, fft_noisy)
plt.xlim(0, 100)
plt.title('FFT Magnitude Spectrum - Noisy ECG')
plt.xlabel('Frequency (Hz)'); plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

wo = 50 / (fs / 2)
Q = 30
b_fixed, a_fixed = iirnotch(wo, Q)
ecg_fixed = filtfilt(b_fixed, a_fixed, ecg_noisy_fixed)

plt.figure(figsize=(12, 5))
plt.plot(time_10s, ecg_fixed[:samples_10s])
plt.title('Task 2: After Fixed 50 Hz Notch Filter (First 10 seconds)')
plt.xlabel('Time (s)'); plt.ylabel('Amplitude (mV)')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

fft_fixed = np.abs(np.fft.rfft(ecg_fixed))
plt.figure(figsize=(12, 5))
plt.plot(f, fft_fixed)
plt.xlim(0, 100)
plt.title('FFT Magnitude Spectrum - After Fixed Notch Filter')
plt.xlabel('Frequency (Hz)'); plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

segment_sec = 5
segment_samples = int(segment_sec * fs)
step = int(segment_samples * 0.5)
ecg_adaptive = np.zeros_like(ecg_noisy_drift)
freq_estimates = []
time_estimates = []

start = 0
while start + segment_samples <= len(ecg_noisy_drift):
    seg = ecg_noisy_drift[start:start + segment_samples]
    fft_seg = np.abs(np.fft.rfft(seg))
    f_seg = np.fft.rfftfreq(len(seg), 1/fs)
    
    range_idx = np.where((f_seg >= 49) & (f_seg <= 51))[0]
    est_freq = 50.0
    if len(range_idx) > 0:
        est_freq = f_seg[range_idx[np.argmax(fft_seg[range_idx])]]
    
    freq_estimates.append(est_freq)
    time_estimates.append((start + segment_samples // 2) / fs)
    
    wo_adapt = est_freq / (fs / 2)
    b_adapt, a_adapt = iirnotch(wo_adapt, Q)
    filtered_seg = filtfilt(b_adapt, a_adapt, seg)
    
    if start == 0:
        ecg_adaptive[start:start + segment_samples] = filtered_seg
    else:
        overlap = segment_samples - step
        fade = np.linspace(1, 0, overlap)
        ecg_adaptive[start:start + overlap] = (
            ecg_adaptive[start:start + overlap] * fade +
            filtered_seg[:overlap] * (1 - fade)
        )
        ecg_adaptive[start + overlap:start + segment_samples] = filtered_seg[overlap:]
    
    start += step

peaks_fixed = pan_tompkins(ecg_fixed)
peaks_adaptive = pan_tompkins(ecg_adaptive)
peaks_noisy = pan_tompkins(ecg_noisy_fixed)

plt.figure(figsize=(14, 12))
plt.suptitle('Pan-Tompkins QRS Detection Results (First 10 seconds)', fontsize=16, y=0.98)  

plt.subplot(2, 1, 1)
plt.plot(time_10s, ecg_noisy_fixed[:samples_10s])
plt.plot(time_10s[peaks_noisy[peaks_noisy < samples_10s]], 
         ecg_noisy_fixed[peaks_noisy[peaks_noisy < samples_10s]], 'ro', markersize=8)
plt.title('With 50 Hz Interference', fontsize=12, pad=20)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.grid(True)
plt.legend(['Signal', 'R-peaks'])

plt.subplot(2, 1, 2)
plt.plot(time_10s, ecg_fixed[:samples_10s])
plt.plot(time_10s[peaks_fixed[peaks_fixed < samples_10s]], 
         ecg_fixed[peaks_fixed[peaks_fixed < samples_10s]], 'ro', markersize=8)
plt.title('After Fixed 50 Hz Notch Filter', fontsize=12, pad=20) 
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.grid(True)
plt.legend(['Signal', 'R-peaks'])

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.show(block=False)

plt.show(block=True)  
