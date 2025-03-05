import wfdb
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import butter, filtfilt
import pywt  # pip install PyWavelets
from scipy.signal import find_peaks
from scipy.fftpack import fft
#import skfuzzy as fuzz
#import skfuzzy.control as ctrl

# 1. Load MIT-BIH dataset
path = './mit/100'
def load_mitbih_record(record_name):
    """Načíta EKG záznam z MIT-BIH databázy"""
    try:
        record = wfdb.rdrecord(f"./mit/{record_name}")
        annotation = wfdb.rdann(f"./mit/{record_name}", 'atr',)
        fs = record.fs  # Vzorkovacia frekvencia
        # vzorky z MLII zvod
        signal = record.p_signal[:, 0]  # MLII zvod

        return signal, fs, annotation
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        return None, None, None

# 2. Noise Filtering
def butterworth_filter(signal, lowcut=0.5, highcut=50.0, fs=360, order=4):
    """Použitie Butterworth filtra na odstránenie šumu"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def wavelet_denoising(signal, wavelet='db4', level=4):
    """Odstránenie šumu pomocou Wavelet Transform"""
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    level = min(level, max_level) if max_level > 0 else 1  # Prevents over-decimation
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745  # Výpočet prahovej hodnoty
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_thresholded, wavelet)

# 3. Signal Segmentation (5-second windows)
def segment_signal(signal, fs, segment_length=5):
    """Rozdelenie signálu na 5-sekundové okná"""
    segment_samples = segment_length * fs
    num_segments = len(signal) // segment_samples
    return np.array_split(signal[:num_segments * segment_samples], num_segments)

# **5. Vylepšená detekcia R-vĺn s adaptívnym prahovaním**
def detect_r_peaks(signal, fs):
    """Detekcia R-vĺn pomocou Pan-Tompkins algoritmu s adaptívnym prahovaním"""
    signals, info = nk.ecg_process(signal, sampling_rate=fs)
    r_peaks = np.array(info["ECG_R_Peaks"])

    # Adaptívne prahovanie: odstránenie slabých detekcií
    threshold = np.mean(signal[r_peaks]) * 0.5  # Nastavenie prahu na 50% priemernej amplitúdy R-vĺn
    r_peaks = r_peaks[signal[r_peaks] > threshold]

    return r_peaks, signals

# **6. Overenie presnosti detekcie R-vĺn oproti MIT-BIH anotáciám**
"""
	•	Butterworth filter (5-15 Hz) → Lepšie zvýraznenie QRS komplexu.
	•	Adaptívne prahovanie v Pan-Tompkins algoritme → Odstránenie slabých detekcií.
	•	Porovnanie s MIT-BIH anotáciami → Vyhodnotenie presnosti:
	•	True Positives (TP) = Správne detegované R-vlny.
	•	False Positives (FP) = Nesprávne detegované vlny.
	•	False Negatives (FN) = Chýbajúce R-vlny.
"""
def compare_r_peaks(r_peaks, annotation):
    """Porovnáva detegované R-vlny s anotáciami MIT-BIH"""
    annotated_peaks = annotation.sample
    true_positives = np.sum(np.isin(r_peaks, annotated_peaks))
    false_positives = len(r_peaks) - true_positives
    false_negatives = len(annotated_peaks) - true_positives

    print(f"✔ True Positives (TP): {true_positives}")
    print(f"❌ False Positives (FP): {false_positives}")
    print(f"❌ False Negatives (FN): {false_negatives}")

    return true_positives, false_positives, false_negatives


##########
"""
    EXTRAKCIA priznakov
"""
def extract_features(signal, r_peaks, fs):
    """Funkcia na extrakciu príznakov zo signálu"""

    # ✅ **1. R-R intervaly (čas medzi R-vlnami)**
    rr_intervals = np.diff(r_peaks) / fs  # Prevod na sekundy

    # ✅ **2. Amplitúda R-vlny**
    r_amplitudes = signal[r_peaks]

    # ✅ **3. Šírka QRS komplexu**
    qrs_widths = []
    for peak in r_peaks:
        left_idx = max(0, peak - int(0.05 * fs))  # ~50 ms pred R-vlnou
        right_idx = min(len(signal), peak + int(0.05 * fs))  # ~50 ms po R-vlne
        qrs_segment = signal[left_idx:right_idx]
        peak_indices, _ = find_peaks(qrs_segment, height=0.3 * np.max(qrs_segment))
        if len(peak_indices) > 0:
            qrs_widths.append(len(qrs_segment) / fs)  # Prevod na sekundy
        else:
            qrs_widths.append(np.nan)

    # ✅ **4. Fourierova transformácia (FFT)**
    freq_spectrum = np.abs(fft(signal))[:len(signal) // 2]  # Používame len prvú polovicu spektra
    max_freq_amplitude = np.max(freq_spectrum)  # Najväčšia amplitúda v spektre

    # ✅ **5. Wavelet transformácia (Časovo-frekvenčné príznaky)**
    coeffs = pywt.wavedec(signal, wavelet='db4', level=4)
    wavelet_energy = np.sum(np.square(coeffs[0]))  # Energia prvej vrstvy

    # ✅ **6. Normalizácia príznakov**
    features = {
        "RR_intervals_mean": np.nanmean(rr_intervals),
        "RR_intervals_std": np.nanstd(rr_intervals),
        "R_amplitude_mean": np.mean(r_amplitudes),
        "R_amplitude_std": np.std(r_amplitudes),
        "QRS_width_mean": np.nanmean(qrs_widths),
        "QRS_width_std": np.nanstd(qrs_widths),
        "Max_FFT_Amplitude": max_freq_amplitude,
        "Wavelet_Energy": wavelet_energy
    }

    return features



def main():
    record_names = ["100", "101", "103", "105", "106"]

    # **Uloženie výsledkov pre každý záznam**
    results = []
    for record_name in record_names:
        raw_signal, fs, annotation = load_mitbih_record(record_name)
        if raw_signal is not None:
            # **Aplikácia Butterworth filtra**
            filtered_signal = butterworth_filter(raw_signal, fs=fs)

            # **Aplikácia Wavelet denoising**
            denoised_signal = wavelet_denoising(filtered_signal)

            # **Odstránenie NaN hodnôt**
            denoised_signal = np.nan_to_num(denoised_signal)

            # **Segmentácia signálu**
            segments = segment_signal(denoised_signal, fs)

            # **Detekcia R-vĺn s adaptívnym prahovaním**
            r_peaks, processed_signals = detect_r_peaks(denoised_signal, fs)

            # **Porovnanie s anotáciami MIT-BIH**
            compare_r_peaks(r_peaks, annotation)

            # **Vizualizácia výsledkov**
            """
            plt.figure(figsize=(12, 4))
            plt.plot(denoised_signal[:3600], label="EKG Signál")
            plt.scatter(r_peaks[r_peaks < 3600], denoised_signal[r_peaks[r_peaks < 3600]], color='red',
                        label="Detegované R-vlny")
            plt.scatter(annotation.sample[annotation.sample < 3600],
                        denoised_signal[annotation.sample[annotation.sample < 3600]], color='green',
                        label="Anotované R-vlny", marker='x')
            plt.title("Predspracovaný EKG signál s porovnaním detegovaných a anotovaných R-vĺn")
            plt.xlabel("Vzorka")
            plt.ylabel("Amplitúda (mV)")
            plt.legend()
            plt.show()
            """
            print("✅ Predspracovanie a detekcia R-vĺn dokončená.")
            # ✅ **Aplikácia extrakcie príznakov na spracovaný signál**
            features = extract_features(denoised_signal, r_peaks, fs)
            print("�� Extrakcia príznakov dokončená.")
            print(features)
            features = extract_features(denoised_signal, r_peaks, fs)
            print(features)


        else:
            print("❌ Error: Failed to load EKG data.")

if __name__ == "__main__":
    main()
# **Zoznam testovacích záznamov z MIT-BIH**

