import matplotlib
import numpy as np
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import wfdb
import pywt
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft

# 🌟 1️⃣ Načítanie EKG signálu (MIT-BIH Arrhythmia Database)
record = wfdb.rdrecord("../mit/100", channels=[0])  # Lead II
fs = record.fs  # Sampling frequency (360 Hz)
ecg_signal = record.p_signal.flatten()[:5 * fs]  # Prvých 5 sekúnd (1800 vzoriek)

# 🌟 2️⃣ Odstránenie baseline driftu pomocou Wavelet Transform (DWT)
wavelet = 'db4'
level = 9  # Počet úrovní decompozície
coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
coeffs[0] = np.zeros_like(coeffs[0])  # Nastavenie najnižšej frekvenčnej zložky na 0 (baseline drift)
ecg_no_baseline = pywt.waverec(coeffs, wavelet)

# 🌟 3️⃣ Odstránenie vysokofrekvenčného šumu pomocou Band-pass filtra (0.5 - 40 Hz)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal, lowcut=0.5, highcut=40, fs=360, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, signal)

ecg_filtered = apply_bandpass_filter(ecg_no_baseline)

# 🌟 4️⃣ Porovnanie signálov: Korelačný koeficient
correlation = np.corrcoef(ecg_signal, ecg_filtered)[0, 1]

# 🌟 5️⃣ FFT analýza signálov (na porovnanie šumu)
def plot_fft(signal, title):
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    spectrum = np.abs(fft(signal))
    plt.plot(freqs[:N // 2], spectrum[:N // 2])
    plt.title(title)
    plt.xlabel("Frekvencia (Hz)")
    plt.ylabel("Amplitúda")

# 🌟 6️⃣ Vizualizácia signálov
plt.figure(figsize=(12, 8))

# Časové priebehy
plt.subplot(3, 1, 1)
plt.plot(ecg_signal, label="Pôvodný EKG signál", color='blue')
plt.legend()
plt.title("Pôvodný EKG signál (5 sekúnd)")

plt.subplot(3, 1, 2)
plt.plot(ecg_no_baseline, label="Po odstránení baseline driftu (Wavelet)", color='green')
plt.legend()
plt.title("Signál po odstránení baseline driftu")

plt.subplot(3, 1, 3)
plt.plot(ecg_filtered, label="Po kombinovanom filtrovaní (DWT + Band-pass)", color='red')
plt.legend()
plt.title(f"Finálne filtrovaný signál (Korelácia: {correlation:.4f})")

plt.tight_layout()
plt.show()

# 🌟 7️⃣ FFT porovnanie
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plot_fft(ecg_signal, "FFT pôvodného signálu")

plt.subplot(2, 1, 2)
plot_fft(ecg_filtered, "FFT po kombinovanom filtrovaní (DWT + Band-pass)")

plt.tight_layout()
plt.show()
