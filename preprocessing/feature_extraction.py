# preprocessing/feature_extraction.py
import numpy as np
import pandas as pd
import pywt
import neurokit2 as nk
from config import QRS_COMPARISON_CSV

"""
HR (Heart Rate):
	•	Počíta sa HR zo vzdialenosti medzi aktuálnym a predošlým R-vrcholom (RR interval).
	•	Overovanie realistické limity pre HR (30 – 220 bpm).

QRS interval:
	•	Používanir neurokit2.ecg_delineate s waveletovou metódou, čo je dobrý prístup na detekciu QRS hraníc.
	•	Kontrola rozsahu platnosti QRS intervalu (40 – 180 ms).

T-vlna amplitude (TWA):
	•	Waveletová rekonštrukcia signálu na úrovni detailu 5 (db4 wavelet).
	•	Vypočítanie TWA ako maximálnu absolútnu amplitúdu vo vhodnom časovom okne po R vrchole (150 – 400 ms).

 Výsledok:
	•	Extrahované príznaky su ukladane do zoznamu a exportovanie aj do CSV na neskoršiu analýzu.

"""


def extract_features_for_fuzzy(signal_for_amplitude, fs, r_peaks):
    try:
        # Delineácia EKG signálu – nájdenie onsets a offsets QRS komplexov
        _, waves_peak = nk.ecg_delineate(signal_for_amplitude, rpeaks=r_peaks, sampling_rate=fs, method="dwt")

    except Exception as e:
        print(f"Error during delineation: {e}")
        return []
    # Načítanie začiatkov a koncov QRS komplexov
    r_onsets = waves_peak.get("ECG_R_Onsets", [None] * len(r_peaks))
    r_offsets = waves_peak.get("ECG_R_Offsets", [None] * len(r_peaks))

    min_len = len(r_peaks)
    features = []
    comparison_log = []

    # Predpočítanie waveletového signálu pre T vlnu alternation (TWA)
    coeffs = pywt.wavedec(signal_for_amplitude, 'db4', level=5)
    selected_coeffs = [None] * len(coeffs)
    selected_coeffs[0] = coeffs[0]
    selected_coeffs[1] = coeffs[1]
    reconstructed = pywt.waverec(selected_coeffs, 'db4')

    # Orezanie výsledného signálu, ak je viac zloziek
    if len(reconstructed) > len(signal_for_amplitude):
        reconstructed = reconstructed[:len(signal_for_amplitude)]

    # Hlavná slučka pre výpočet príznakov z každého úderu
    for i in range(1, min_len):
        # RR interval a HR
        rr_interval = (r_peaks[i] - r_peaks[i - 1]) / fs
        if rr_interval <= 0:
            continue
        hr = 60 / rr_interval
        if not (30 <= hr <= 220):
            continue

        # Overenie QRS komplexu
        ro, rf = r_onsets[i], r_offsets[i]
        if ro is None or rf is None or rf <= ro:
            continue
        qrs_rorr = (rf - ro) / fs * 1000
        if not (40 <= qrs_rorr <= 180):
            continue

        # # Hľadanie okna pre T vlnu alternativna analýz
        r = r_peaks[i]
        search_start = r + int(0.15 * fs)
        search_end = r + int(0.4 * fs)
        if search_end >= len(reconstructed) or (i + 1 < len(r_peaks) and search_end > r_peaks[i + 1]):
            continue
        twa_wavelet = max(np.abs(reconstructed[search_start:search_end]))
        if not (0 < twa_wavelet <= 1.0):
            continue

        # ukladanie validných dát

        comparison_log.append({"Index": i, "HR": hr, "QRS_rorr": qrs_rorr, "TWA": twa_wavelet})
        features.append({"Index": int(i), "HR": hr, "QRS_interval": qrs_rorr, "T_wave": twa_wavelet})

    # Export výsledkov porovnania QRS a TWA do CSV
    df_cmp = pd.DataFrame(comparison_log)
    QRS_COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_cmp.to_csv(QRS_COMPARISON_CSV, index=False)
    return features
