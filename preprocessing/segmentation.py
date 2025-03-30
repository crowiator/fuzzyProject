# preprocessing/segmentation.py
import numpy as np


# Segmentacia
# ➡ Pre každú R-vlnu vyberieme segment signálu od R-0.2s do R+0.4s.
# Vytvoríme pole segmentov, kde každý segment má rovnakú dĺžku.
def segment_heartbeats(signal, r_peaks, fs, pre_R=0.2, post_R=0.4):
    """
    Segmentuje EKG signál na jednotlivé údery srdca okolo R-vĺn.

    Výstup:
        segments (np.array): shape = (N, segment_length, 1)
    """
    pre_samples = int(pre_R * fs)
    post_samples = int(post_R * fs)
    segments = []

    for r in r_peaks:
        if r - pre_samples >= 0 and r + post_samples < len(signal):
            segment = signal[r - pre_samples:r + post_samples]
            segments.append(segment)

    segments = np.array(segments)
    return np.expand_dims(segments, axis=-1)  # pre 1D CNN
