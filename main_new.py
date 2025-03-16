import matplotlib
import numpy as np
import wfdb
import pywt
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import butter, filtfilt
matplotlib.use("TkAgg")

# 1️⃣ Function to Load ECG Signal
def load_mitbih_record(record_name, path="./mit/"):
    """Load ECG signal and annotations from MIT-BIH database."""
    try:
        record = wfdb.rdrecord(f"{path}{record_name}")
        annotation = wfdb.rdann(f"{path}{record_name}", 'atr')
        fs = record.fs
        signal = record.p_signal[:, 0]

        r_peak_positions = annotation.sample
        beat_types = annotation.symbol
        beat_counts = Counter(beat_types)

        return signal, fs, r_peak_positions, beat_counts
    except Exception as e:
        print(f"❌ Error loading record {record_name}: {e}")
        return None, None, None, None

# 2️⃣ DWT-Based Baseline Drift Removal
import numpy as np
import pywt

def dwt_filtering(ecg_signal, wavelet='db4', level=9, threshold_factor=0.2):
    """
    Aplikuje DWT-based filtering s miernym oslabením detailných koeficientov.
    """
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)

    # Výpočet univerzálneho prahu (Donoho)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = threshold_factor * sigma

    # Soft thresholding pre aproximáciu
    coeffs[0] = pywt.threshold(coeffs[0], threshold, mode='soft')

    # Namiesto úplného odstránenia detailov ich oslabíme len o 30% (miernejšia filtrácia)
    for i in range(1, 4):
        coeffs[i] *= 0.7

    # Rekonštrukcia signálu pomocou IDWT
    filtered_signal = pywt.waverec(coeffs, wavelet)

    return filtered_signal[:len(ecg_signal)]

def lowpass_filter(signal, fs, cutoff=30, order=4):
    """
    Apply a Butterworth low-pass filter to remove high-frequency noise.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


# 3️⃣ Visualization Function
def show_plot_ecg_signal(raw_signal, filtered_signal, fs, duration):
    """Plots raw and filtered ECG signals."""
    num_samples = min(len(raw_signal), len(filtered_signal), int(fs * duration))
    time_axis = np.arange(num_samples) / fs

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, raw_signal[:num_samples], label="Original Signal", alpha=0.7, color='red')
    plt.plot(time_axis, filtered_signal[:num_samples], label="Filtered Signal (DWT)", linewidth=2, color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Baseline Drift and Noise Removal in ECG Signal")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"📊 Mean (Original): {np.mean(raw_signal):.5f}")
    print(f"📊 Mean (Filtered): {np.mean(filtered_signal):.5f}")
    print(f"🔺 Min/Max (Original): {np.min(raw_signal):.5f}, {np.max(raw_signal):.5f}")
    print(f"🔺 Min/Max (Filtered): {np.min(filtered_signal):.5f}, {np.max(filtered_signal):.5f}")
    print(f"🔹 Variance (Original): {np.var(raw_signal):.5f}")
    print(f"🔹 Variance (Filtered): {np.var(filtered_signal):.5f}")

    # Ensure equal lengths before computing correlation
    min_len = min(len(raw_signal), len(filtered_signal))
    correlation = np.corrcoef(raw_signal[:min_len], filtered_signal[:min_len])[0, 1]
    print(f"🔗 Correlation between Original and Filtered Signal: {correlation:.5f}")

def normalize_signal(filtered_signal, original_signal):
    """
    Normalize the amplitude of the filtered signal to match the original signal.
    """
    return (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal) * np.std(original_signal) + np.mean(original_signal)


# 4️⃣ Main Execution
if __name__ == "__main__":
    raw_signal, fs, _, _ = load_mitbih_record("100")

    # 1. Najskôr odstránime vysokofrekvenčný šum
    filtered_signal = lowpass_filter(raw_signal, fs, cutoff=35)

    # 2. Potom odstránime baseline drift pomocou DWT
    filtered_signal = dwt_filtering(filtered_signal, threshold_factor=0.15)

    # Step 3: Normalize to match original amplitude
    filtered_signal = normalize_signal(filtered_signal, raw_signal)

    # Step 4: Plot both signals
    show_plot_ecg_signal(raw_signal, filtered_signal, fs, duration=10)
