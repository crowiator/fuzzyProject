import matplotlib
import numpy as np
import wfdb
import pandas as pd
matplotlib.use("TkAgg")


import numpy as np
import scipy.signal as signal
import pywt
import wfdb
import matplotlib.pyplot as plt

def load_mitbih_record(record_name, path="./mit/"):
    """Načíta EKG signál z MIT-BIH databázy."""
    try:
        record = wfdb.rdrecord(f"{path}{record_name}")
        annotation = wfdb.rdann(f"{path}{record_name}", 'atr')
        fs = record.fs
        signal = record.p_signal[:, 0]
        return signal, fs
    except Exception as e:
        print(f"❌ Chyba pri načítaní záznamu {record_name}: {e}")
        return None, None

def butterworth_highpass(ecg_signal, fs, cutoff=0.5, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    return filtered_signal

def wavelet_baseline_correction(ecg_signal, wavelet='db3', level=7):
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])  # Odstránenie baseline komponenty
    corrected_signal = pywt.waverec(coeffs, wavelet)
    return corrected_signal[:len(ecg_signal)]

def compute_snr(original, denoised):
    noise = original - denoised
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_prd(original, denoised):
    prd = np.sqrt(np.sum((original - denoised) ** 2) / np.sum(original ** 2)) * 100
    return prd

# Zoznam EKG záznamov na analýzu
records = [str(i) for i in range(100, 124)]  # Pridaj ďalšie záznamy podľa potreby

# Uloženie výsledkov
snr_butter_list, prd_butter_list = [], []
snr_wavelet_list, prd_wavelet_list = [], []

for record_name in records:
    ecg_signal, fs = load_mitbih_record(record_name)
    if ecg_signal is None:
        continue  # Preskočíme chybné načítania

    # Aplikácia filtrov
    butter_filtered = butterworth_highpass(ecg_signal, fs)
    wavelet_filtered = wavelet_baseline_correction(ecg_signal)

    # Výpočet metrik
    snr_butter = compute_snr(ecg_signal, butter_filtered)
    prd_butter = compute_prd(ecg_signal, butter_filtered)
    snr_wavelet = compute_snr(ecg_signal, wavelet_filtered)
    prd_wavelet = compute_prd(ecg_signal, wavelet_filtered)

    # Uloženie výsledkov
    snr_butter_list.append(snr_butter)
    prd_butter_list.append(prd_butter)
    snr_wavelet_list.append(snr_wavelet)
    prd_wavelet_list.append(prd_wavelet)

    print(f"📊 Record {record_name}:")
    print(f"  Butterworth -> SNR: {snr_butter:.2f} dB, PRD: {prd_butter:.2f} %")
    print(f"  Wavelet Correction -> SNR: {snr_wavelet:.2f} dB, PRD: {prd_wavelet:.2f} %")

# Výpočet priemerných hodnôt pre všetky záznamy
avg_snr_butter = np.mean(snr_butter_list)
avg_prd_butter = np.mean(prd_butter_list)
avg_snr_wavelet = np.mean(snr_wavelet_list)
avg_prd_wavelet = np.mean(prd_wavelet_list)

print("\n🔎 **Zhrnutie analýzy pre všetky záznamy:**")
print(f"Butterworth Filter -> Priemerné SNR: {avg_snr_butter:.2f} dB, Priemerné PRD: {avg_prd_butter:.2f} %")
print(f"Wavelet Correction -> Priemerné SNR: {avg_snr_wavelet:.2f} dB, Priemerné PRD: {avg_prd_wavelet:.2f} %")

# Porovnanie metód graficky
labels = ["Butterworth", "Wavelet"]
snr_values = [avg_snr_butter, avg_snr_wavelet]
prd_values = [avg_prd_butter, avg_prd_wavelet]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(labels, snr_values, color=['blue', 'red'])
plt.ylabel("Priemerné SNR (dB)")
plt.title("Porovnanie SNR")

plt.subplot(1, 2, 2)
plt.bar(labels, prd_values, color=['blue', 'red'])
plt.ylabel("Priemerné PRD (%)")
plt.title("Porovnanie PRD")

plt.show()
