#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Return T-wave peak indices, times, and amplitudes from a MIT-BIH record.
Requires: neurokit2 ≥ 0.2.10, wfdb, pandas, numpy.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import matplotlib
from config import ALL_RECORDS, LEAD, OUT_DIR, FEAT_DIR

matplotlib.use('TkAgg')
import wfdb
import neurokit2 as nk
import pandas as pd
import numpy as np



# --------------------------- DEMO ------------------------------------ #
if __name__ == "__main__":
    print("start")
    path = "data/mit/100"
    rec = wfdb.rdrecord(path)
    print(dir(rec))
    print(rec.baseline)
    fs = rec.fs
    try:
        sig_idx = rec.sig_name.index("MLII")
    except ValueError as err:
        raise ValueError(f"Lead 100 not found. Available: {rec.sig_name}") from err
    print("asdadasdasdasd")
    ecg_raw = rec.p_signal[:, sig_idx]
     # presne podľa WFDB definície
    # Extract R-peaks locations
    print("asdadasdasdasd")


    clean_signal = nk.ecg_clean(ecg_raw, sampling_rate=fs, method="neurokit")
    # finálne nulovanie (odstráni zvyšný posun a udrží vlny kladné)

    # Vizualizácia porovnania signálov
    plt.figure(figsize=(15, 5))
    plt.plot(ecg_raw[:2000], label='Raw ECG', alpha=0.6)
    plt.plot(clean_signal[:2000], label='Cleaned ECG (NeuroKit)', linewidth=1.5)
    plt.legend()
    plt.grid()
    plt.title("Baseline drift correction")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()
    _, rpeaks = nk.ecg_peaks(ecg_raw, sampling_rate=360)
    print(rpeaks)
    signal_cwt, waves_cwt = nk.ecg_delineate(ecg_raw,
                                         rpeaks['ECG_R_Peaks'],
                                         sampling_rate=360,
                                         method="dwt",
                                         show=False)
    # Najprv získaš indexy T-vĺn
    t_peaks_idx = waves_cwt["ECG_T_Peaks"]

    # Odstráň hodnoty NaN (ak neboli všetky vlny detegované)
    t_peaks_idx_clean = [idx for idx in t_peaks_idx if not np.isnan(idx)]

    # Konverzia indexov na celé čísla
    t_peaks_idx_clean = np.array(t_peaks_idx_clean, dtype=int)

    # Získanie amplitúd T-vĺn zo surového signálu
    t_peaks_amplitudes = ecg_raw[t_peaks_idx_clean]

    # QRS intervals
    qrs_onsets = waves_cwt["ECG_R_Onsets"]
    qrs_offsets = waves_cwt["ECG_R_Offsets"]

    qrs_intervals = []
    valid_indices = []

    for onset, offset in zip(qrs_onsets, qrs_offsets):
        if not np.isnan(onset) and not np.isnan(offset):
            interval_duration = (offset - onset) / fs
            qrs_intervals.append(interval_duration)
            valid_indices.append(int(onset))
    print(len(qrs_intervals))
    print(len(t_peaks_amplitudes))
    # Príprava dát do DataFrame
    df_t_waves = pd.DataFrame({
        "Amplitude": t_peaks_amplitudes,
        "QRS_Duration_s": qrs_intervals
    })

    # Ulož do CSV súboru
    output_csv = "t_wave_features.csv"
    df_t_waves.to_csv(output_csv, index=False)

    print(f"Výsledky boli uložené do {output_csv}")

    # Visualization including T-waves
    plt.figure(figsize=(15, 5))
    plt.plot(ecg_raw[:2000], label='Raw ECG', alpha=0.6)
    plt.plot(clean_signal[:2000], label='Cleaned ECG (NeuroKit)', linewidth=1.5)

    # Mark T-wave peaks
    t_peaks_display = [idx for idx in t_peaks_idx_clean if idx < 2000]
    plt.scatter(t_peaks_display, clean_signal[t_peaks_display], color='red', label='T-wave peaks', zorder=5)

    plt.legend()
    plt.grid()
    plt.title("Baseline drift correction with T-wave peaks")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()


