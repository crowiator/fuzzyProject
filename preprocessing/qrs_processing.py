# preprocessing/qrs_processing.py
import numpy as np
from scipy.signal import find_peaks
def differentiate(signal):
    """
    Aplikuje diferenciáciu na zvýraznenie rýchlych zmien v signáli.
    """
    diff_signal = np.diff(signal)
    return np.append(diff_signal, 0)


def squaring(signal):
    """
    Umocní signál na zvýraznenie veľkých hodnôt a potlačenie malých.
    """
    return np.power(signal, 2)


def moving_window_integration(signal, window_size=20):
    """
    Aplikuje klznú integráciu na vyhladenie signálu.
    """
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

def detect_r_peaks(integrated_signal, fs, threshold_factor=0.4):
    """
    Deteguje R-vrcholy pomocou adaptívneho prahovania a find_peaks.
    """
    threshold = threshold_factor * np.max(integrated_signal)

    # Použijeme find_peaks() pre lepšiu detekciu lokálnych maxím
    peaks, _ = find_peaks(integrated_signal, height=threshold, distance=int(0.2 * fs), prominence=0.01)

    return peaks


def find_matching_peaks(annotated_peaks, detected_peaks, fs, tolerance=50):
    """
    Porovná anotované a detegované R-vrcholy s toleranciou v milisekundách.
    """
    # /1000 lebo prevadzam na milisekundy
    tolerance_samples = int((tolerance / 1000) * fs)  # Konverzia na počet vzoriek

    matched = []
    for peak in detected_peaks:
        if np.any(np.abs(annotated_peaks - peak) <= tolerance_samples):
            matched.append(peak)

    return np.array(matched)
