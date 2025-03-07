import wfdb
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pywt
import neurokit2 as nk
import pandas as pd
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

# Funkcia na detekciu R-vĺn pomocou Pan-Tompkins algoritmu (NeuroKit2)
def detect_r_peaks(signal, fs):
    """
    Vykoná detekciu R-vĺn v EKG signáli pomocou Pan-Tompkins algoritmu.

    Parametre:
    signal (array): filtrovaný a odšumený EKG signál
    fs (int): vzorkovacia frekvencia signálu

    Výstup:
    r_peaks (array): indexy detegovaných R-vĺn
    processed_signals (DataFrame): signál so spracovaním (QRS komplexy, T/P vlny)
    """
    processed_signals, info = nk.ecg_process(signal, sampling_rate=fs)
    r_peaks = info["ECG_R_Peaks"]
    return r_peaks, processed_signals


# Funkcia na vizualizáciu výsledkov detekcie R-vĺn
def plot_detected_r_peaks(signal, r_peaks, fs, record_name, num_samples=2000):
    """
    Vykreslí EKG signál s detegovanými R-vlnami.

    Parametre:
    signal (array): filtrovaný EKG signál
    r_peaks (array): indexy detegovaných R-vĺn
    fs (int): vzorkovacia frekvencia
    record_name (str): názov EKG záznamu
    num_samples (int): počet vzoriek na zobrazenie v grafe
    """
    plt.figure(figsize=(12, 4))

    # Vykreslenie signálu
    plt.plot(signal[:num_samples], label="Filtrovaný EKG signál", color="navy")

    # Vykreslenie detegovaných R-vĺn
    plt.scatter(r_peaks[r_peaks < num_samples],
                signal[r_peaks[r_peaks < num_samples]],
                color='red', marker='o', label="Detegované R-vlny")

    plt.xlabel("Vzorky")
    plt.ylabel("Amplitúda")
    plt.title(f"EKG signál a detegované R-vlny (Pan-Tompkins) - {record_name}")
    plt.legend()
    plt.show()
# Funkcia na vyhodnotenie presnosti detekcie voči MIT-BIH anotáciám
def evaluate_detection(r_peaks_detected, r_peaks_true, tolerance=5):
    """
    Porovná detegované R-vlny s anotáciami z MIT-BIH databázy a vypočíta TP, FP, FN.

    Parametre:
    r_peaks_detected (array): detegované R-vlny
    r_peaks_true (array): skutočné (anotované) R-vlny z databázy
    tolerance (int): povolená odchýlka detekcie v počte vzoriek

    Výstup:
    Vypíše počet True Positives, False Positives a False Negatives
    """
    true_positives = sum([1 for peak in r_peaks_detected if any(abs(r_peaks_true - peak) <= tolerance)])
    false_positives = len(r_peaks_detected) - true_positives
    false_negatives = len(r_peaks_true) - true_positives

    print(f"✔️ True Positives (TP): {true_positives}")
    print(f"❌ False Positives (FP): {false_positives}")
    print(f"❌ False Negatives (FN): {false_negatives}")

    # Funkcia na segmentáciu EKG signálu


def segment_ecg(signal, r_peaks, fs, pre_window=0.2, post_window=0.4):
    """
       Rozdelí EKG signál do segmentov (beats) na základe pozícií R-vĺn.

       Parametre:
       signal (array): filtrovaný EKG signál
       r_peaks (array): indexy detegovaných R-vĺn
       fs (int): vzorkovacia frekvencia
       pre_window (float): čas v sekundách pred R-peakom
       post_window (float): čas v sekundách po R-peaku

       Výstup:
       segments (list): zoznam segmentov (beats)
       """
    segments = []
    pre_samples = int(pre_window * fs)  # vzorky pred R-peakom
    post_samples = int(post_window * fs)  # vzorky po R-peaku

    for r in r_peaks:
        # Zaistíme, že segmenty nepresahujú hranice signálu
        start = max(r - pre_samples, 0)
        end = min(r + post_samples, len(signal))

        # extrahujeme segment a pridáme do zoznamu
        segment = signal[start:end]
        segments.append(segment)

    return segments

    return segments


# Funkcia na extrakciu príznakov zo segmentov EKG signálu
def extract_features(segments, fs):
    """
    Extrahuje časové, frekvenčné a waveletové príznaky zo segmentov EKG signálov.

    Parametre:
    segments (list): zoznam EKG segmentov
    fs (int): vzorkovacia frekvencia

    Výstup:
    features_list (list): zoznam slovníkov s príznakmi pre každý segment
    """
    features_list = []

    for segment in segments:
        features = {}

        # Časové príznaky
        features_r_amplitude = np.max(segment)
        features_qrs_duration = np.ptp(segment)  # Peak-to-peak duration

        # Frekvenčné príznaky (FFT)
        freq_spectrum = np.abs(np.fft.fft(segment))[:len(segment)//2]
        dominant_freq = np.argmax(freq_spectrum) * fs / len(segment)
        fft_energy = np.sum(freq_spectrum**2)

        # Waveletové príznaky (DWT)
        coeffs = pywt.wavedec(segment, 'db4', level=4)
        wavelet_energy = [np.sum(np.square(c)) for c in coeffs]

        features = {
            'R_amplitude': features_qrs_duration,
            'QRS_duration': features_qrs_duration,
            'Dominant_frequency': dominant_freq,
            'FFT_energy': np.sum(freq_spectrum**2),
            'Wavelet_energy_approx': wavelet_energy[0],
            'Wavelet_energy_detail1': wavelet_energy[1],
            'Wavelet_energy_detail2': wavelet_energy[1],
            'Wavelet_energy_detail_level2': wavelet_energy[2],
            'Wavelet_energy_detail_level3': wavelet_energy[3]
        }

        features_list.append(features)

    return features_list

# 4️⃣ Hlavná časť programu
if __name__ == "__main__":
    record_name = "100"
    raw_signal, fs, r_peaks_true, beat_types = load_mitbih_record(record_name)

    if raw_signal is not None:
        # Odstránenie baseline driftu
        # 1. Odstránenie baseline driftu
        filtered_signal = butter_highpass(raw_signal, fs=fs)

        # 2. Odstránenie vysokofrekvenčného šumu pomocou DWT
        denoised_signal = wavelet_denoising(signal=filtered_signal, wavelet='db4', level=4)

        # Detekcia R-vĺn pomocou Pan-Tompkins algoritmu
        r_peaks_detected, processed_signals = detect_r_peaks(denoised_signal, fs)

        # Vizualizácia výsledkov
        plot_detected_r_peaks(denoised_signal, r_peaks_detected, fs, record_name)
        print("some tu")
        # Segmentácia signálu
        segments = segment_ecg(denoised_signal, r_peaks_detected, fs)
        print(segments)
        # Vyhodnotenie presnosti voči anotáciám
        evaluate_detection(r_peaks_detected, r_peaks_true, tolerance=5)
        # Extrakcia príznakov zo segmentov
        print(extract_features(segments,fs))
        #compare_signals(filtered_signal, denoised_signal, r_peaks, record_name)
