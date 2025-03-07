import wfdb
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pywt
# 1️⃣ Funkcia na načítanie EKG signálu a anotácií
def load_mitbih_record(record_name, path="./mit/"):
    """Načíta EKG signál a anotácie z MIT-BIH databázy."""
    try:
        record = wfdb.rdrecord(f"{path}{record_name}")
        annotation = wfdb.rdann(f"{path}{record_name}", 'atr')
        fs = record.fs
        signal = record.p_signal[:, 0]

        # Extrahujeme R-peaky a kategórie tepov
        r_peak_positions = annotation.sample
        beat_types = annotation.symbol

        return signal, fs, r_peak_positions, beat_types
    except Exception as e:
        print(f"❌ Chyba pri načítaní záznamu {record_name}: {e}")
        return None, None, None, None

# 2️⃣ Funkcia na odstránenie baseline driftu (Butterworth High-Pass Filter)
def butter_highpass(signal, cutoff=0.5, fs=360, order=4):
    """Aplikuje Butterworth High-Pass Filter na EKG signál."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def wavelet_denoising(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-level])) / 0.6745
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)

    # Kontrola dĺžky
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    elif len(denoised_signal) < len(signal):
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), 'constant')

    return denoised_signal

# 3️⃣ Funkcia na vykreslenie porovnania surového a filtrovaného signálu
def compare_signals(filtered_signal, denoised_signal, r_peaks, record_name, num_samples=1000):
    plt.figure(figsize=(12, 5))

    # 🔹 1. Graf - Butterworth filter (baseline drift odstránený)
    plt.subplot(1, 2, 1)
    plt.plot(filtered_signal[:num_samples], label="Po Butterworth filtre", color="b")
    plt.scatter(r_peaks[r_peaks < num_samples], filtered_signal[r_peaks[r_peaks < num_samples]], color="r", marker='o', label="R-peaky")
    plt.xlabel("Vzorky")
    plt.ylabel("Amplitúda")
    plt.title(f"EKG po Butterworth filtre - {record_name}")
    plt.legend()

    # 🔹 2. Graf - Po Waveletovej filtrácii (DWT)
    plt.subplot(1, 2, 2)
    plt.plot(denoised_signal[:num_samples], label="Po DWT filtrácii", color="g")
    plt.scatter(r_peaks[r_peaks < num_samples], denoised_signal[r_peaks[r_peaks < num_samples]], color="r", marker='o', label="R-peaky")
    plt.xlabel("Vzorky")
    plt.ylabel("Amplitúda")
    plt.title(f"EKG po Wavelet filtrácii (DWT) - {record_name}")
    plt.legend()

    plt.tight_layout()
    plt.show()



# 4️⃣ Hlavná časť programu
if __name__ == "__main__":
    record_name = "100"
    raw_signal, fs, r_peaks, beat_types = load_mitbih_record(record_name)

    if raw_signal is not None:
        # Odstránenie baseline driftu
        # 1. Odstránenie baseline driftu
        filtered_signal = butter_highpass(raw_signal, fs=fs)

        # 2. Odstránenie vysokofrekvenčného šumu pomocou DWT
        denoised_signal = wavelet_denoising(signal=filtered_signal, wavelet='db4', level=4)

    compare_signals(filtered_signal, denoised_signal, r_peaks, record_name)
