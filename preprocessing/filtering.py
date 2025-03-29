# preprocessing/filtering.py
import numpy as np
import pywt
from scipy.signal import butter, filtfilt


def lowpass_filter(signal, fs, cutoff=30, order=4):
    """
    Butterworth low-pass filter na odstránenie vysokofrekvenčného šumu.
    Používa Butterworthov dolnopriepustný filter, ktorý prepustí len frekvencie nižšie ako 30 Hz.

🔍 Na čo je dobrý:
	•	Odstraňuje vysokofrekvenčný šum (napr. EMG – svalový šum, elektrické rušenie).
	•	Zachováva T-vlny (1–10 Hz) aj QRS komplexy (~15 Hz).
	Order 4 je štandardné odporúčanie v kardiologickom signálnom spracovaní,
pretože poskytuje dobrú potlačenie šumu bez narušenia EKG morfológie.
    """
    #Nyquistova frekvencia = polovica vzorkovacej frekvencie fs
    #Aby si správne zachytil (zrekonštruoval) signál, musíš ho vzorkovať aspoň 2× rýchlejšie, než je jeho najvyššia frekvencia.
    nyquist = 0.5 * fs
    #Digitálne filtre (napr. butter()) nechcú prah v Hz, ale v pomere k Nyquistovej frekvencii
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def dwt_filtering(ecg_signal, wavelet='db4', threshold_factor=0.2):
    """
    DWT-based baseline wander removal s mäkkým thresholdingom.
    Používa automatickú detekciu maximálnej úrovne (max level = 5).
    """
    level = pywt.dwt_max_level(len(ecg_signal), pywt.Wavelet('db4').dec_len)
    level = min(level, 5)
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


def normalize_zscore(signal):
    """
        Z-skórová normalizácia EKG signálu.
    """
    if np.std(signal) == 0:
        return signal
    return (signal - np.mean(signal)) / np.std(signal)


def normalize_minmax(signal, range_min=-1, range_max=1):
    """
        Normalizácia signálu do zadaného rozsahu (default: -1 až 1).
    """
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * (range_max - range_min) + range_min


# ----------------------------------------------------------------

