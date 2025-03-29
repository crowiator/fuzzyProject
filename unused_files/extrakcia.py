import numpy as np
import pandas as pd
import scipy.signal as signal
import pywt
import neurokit2 as nk
import wfdb

# Načítanie dát (MIT-BIH databáza)
import wfdb

def load_mitbih_record(record_name, channel=0):
    record = wfdb.rdrecord(record_name)
    signal = record.p_signal[:, channel]
    fs = record.fs
    return signal, fs

# 1. Butterworthov vysokopriepustný filter na baseline wander
def highpass_filter(signal_raw, fs, cutoff=0.5):
    b, a = signal.butter(4, cutoff / (fs / 2), 'high')
    filtered_signal = signal.filtfilt(b, a, signal)
    return filtered_signal

# 2. Wavelet Denoising
def wavelet_denoise(signal, wavelet='db4', level=4):
    import pywt
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal

# 3. Normalizácia (Z-score)
def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

# 4. Detekcia QRS komplexu pomocou Pan-Tompkins (implementácia pomocou NeuroKit2)
def detect_qrs(signal, fs):
    import neurokit2 as nk
    _, rpeaks = nk.ecg_peaks(signal, fs)
    return rpeaks['ECG_R_Peaks']

# 5. Výpočet RR intervalov
def calculate_rr_intervals(r_peaks, fs):
    rr_intervals = np.diff(r_peaks) / fs
    RR0 = np.insert(rr_intervals, 0, np.nan)
    RR1 = np.append(rr_intervals, np.nan)
    return RR0, RR1

# 6. Jednoduchá detekcia amplitúdy R-vlny
def get_r_amplitudes(signal, r_peaks):
    return signal[r_peaks]

# Ukážkové spustenie (príklad)
import numpy as np
import pandas as pd

record_name = './mit/100'  # Príklad názvu záznamu z MIT-BIH databázy
signal, fs = load_mitbih_record(record_name)

# Predspracovanie
filtered_signal = highpass_filter(signal, fs, 0.5)
denoised_signal = wavelet_denoise(filtered_signal)
normalized_signal = normalize(denoised_signal)

# Detekcia QRS
r_peaks = detect_qrs(normalize(signal), fs)

# RR intervaly
RR0 = calculate_rr_intervals(r_peaks, fs)
RR1 = np.roll(RR0, -1)

# Amplitúda R vlny
R_amp = get_r_amplitudes(normalize(signal), r_peaks)

# Vytvorenie dátového rámca pre fuzzy logiku
# Pozn: Detekciu P a T vĺn a QRS polaritu treba implementovať pokročilejšie (zatiaľ ukážkové hodnoty)

df = pd.DataFrame({
    'RR0': RR0,
    'RR1': RR1,
    'P_wave': np.zeros(len(r_peaks)),  # placeholder: 0 (neprítomná), 1 (pozitívna), -1 (negatívna)
    'QRS_complex': np.ones(len(r_peaks)),  # placeholder
    'R_amplitude': R_amp,
    'T_wave': np.zeros(len(r_peaks)),  # placeholder
})

print(df.head())