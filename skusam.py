

import numpy as np
import wfdb
import neurokit2 as nk
from scipy.signal import butter, filtfilt
import pywt

# Načítanie EKG signálu z MIT-BIH databázy
def load_mitbih_record(record_name, path="./mit/"):
    record = wfdb.rdrecord(f"{path}{record_name}")
    annotation = wfdb.rdann(f"{path}{record_name}", 'atr')
    fs = record.fs
    signal = record.p_signal[:, 0]
    return signal, fs, annotation.sample

# Butterworth High-Pass Filter na odstránenie baseline driftu
def butter_highpass(signal, cutoff=0.5, fs=360, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return filtfilt(b, a, signal)

# Wavelet denoising na odstránenie vysokofrekvenčného šumu
def wavelet_denoising(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-level])) / 0.6745
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)[:len(signal)]

# Extrakcia príznakov
def extract_ecg_features(signal, fs):
    r_peaks_dict, _ = nk.ecg_peaks(signal, sampling_rate=fs)
    r_peaks = np.unique(r_peaks_dict["ECG_R_Peaks"])

    # Menej agresívne odstraňovanie krátkych intervalov (RR < 0.15 s)
    rr_intervals = np.diff(r_peaks) / fs
    valid_indices = np.where(rr_intervals > 0.15)[0] + 1
    r_peaks = np.insert(r_peaks[valid_indices], 0, r_peaks[0])

    if len(r_peaks) < 3:
        raise ValueError("Not enough valid R-peaks for feature extraction.")

    # Delineácia (P, QRS, T vlny)
    delineate, _ = nk.ecg_delineate(signal, r_peaks, sampling_rate=fs, method="dwt")

    # RR intervaly
    rr_intervals = np.diff(r_peaks) / fs
    RR_0 = np.insert(rr_intervals, 0, np.nan)
    RR_1 = np.append(rr_intervals, np.nan)

    # QRS šírka
    q_peaks = delineate["ECG_Q_Peaks"]
    s_peaks = delineate["ECG_S_Peaks"]
    qrs_width = np.nanmean((s_peaks - q_peaks) / fs)

    # Amplitúda R vlny
    r_amplitude = np.mean(signal[r_peaks])

    # P vlna
    p_peaks = delineate["ECG_P_Peaks"]
    p_onsets = delineate["ECG_P_Onsets"]
    p_offsets = delineate["ECG_P_Offsets"]
    p_amplitude = np.nanmean(signal[p_peaks])
    p_duration = np.nanmean((p_offsets - p_onsets) / fs)

    # T vlna
    t_peaks = delineate["ECG_T_Peaks"]
    t_onsets = delineate["ECG_T_Onsets"]
    t_offsets = delineate["ECG_T_Offsets"]
    t_amplitude = np.nanmean(signal[t_peaks])
    t_duration = np.nanmean((t_offsets - t_onsets) / fs)

    features = {
        "RR_previous": RR_0,
        "RR_next": RR_1,
        "QRS_width": qrs_width,
        "R_amplitude": r_amplitude,
        "P_amplitude": p_amplitude,
        "P_duration": p_duration,
        "T_amplitude": t_amplitude,
        "T_duration": t_duration
    }

    return features

# Spustenie celého postupu
if __name__ == "__main__":
    record_name = "100"
    raw_signal, fs, _ = load_mitbih_record(record_name)

    filtered_signal = butter_highpass(raw_signal, fs=fs)
    denoised_signal = wavelet_denoising(filtered_signal)

    features = extract_ecg_features(denoised_signal, fs)

    for key, value in features.items():
        print(f"{key}: {value}")
