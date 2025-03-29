# preprocessing/segmentation.py
import numpy as np


# Segmentacia
# ➡ Pre každú R-vlnu vyberieme segment signálu od R-0.2s do R+0.4s.
# Vytvoríme pole segmentov, kde každý segment má rovnakú dĺžku.
def segment_heartbeats(signal, r_peaks, fs, pre_R=0.2, post_R=0.4):
    """
    Segmentuje EKG signál na jednotlivé údery srdca okolo detegovaných R-vĺn.

    Parametre:
        signal (np.array): EKG signál
        r_peaks (list): indexy R-vrcholu
        fs (int): vzorkovacia frekvencia (Hz)
        pre_R (float): trvanie segmentu pred R-vrcholom v sekundách
        post_R (float): trvanie segmentu po R-vrchole v sekundách

    Výstup:
        segments (np.array): zoznam segmentov s jednotnou dĺžkou
    """
    pre_samples = int(pre_R * fs)  # Počet vzoriek pred R
    post_samples = int(post_R * fs)  # Počet vzoriek po R

    segments = []
    for r in r_peaks:
        if r - pre_samples >= 0 and r + post_samples < len(signal):
            segment = signal[r - pre_samples:r + post_samples]
            segments.append(segment)

    return np.array(segments)
