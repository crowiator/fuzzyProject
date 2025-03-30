import numpy as np
import pandas as pd
import pywt
import neurokit2 as nk
from config import QRS_COMPARISON_CSV
def extract_features_for_fuzzy(signal_for_amplitude, fs, r_peaks):

    try:
        _, waves_peak = nk.ecg_delineate(signal_for_amplitude, rpeaks=r_peaks, sampling_rate=fs, method="dwt")
        #print(waves_peak.keys())
    except Exception as e:
        print(f"❌ Chyba pri delineácii: {e}")
        return []

    r_onsets = waves_peak.get("ECG_R_Onsets", [None] * len(r_peaks))
    r_offsets = waves_peak.get("ECG_R_Offsets", [None] * len(r_peaks))

    min_len = len(r_peaks)
    features = []
    comparison_log = []

    # Predpočítaj waveletový signál pre TWA
    coeffs = pywt.wavedec(signal_for_amplitude, 'db4', level=5)
    selected_coeffs = [None] * len(coeffs)
    selected_coeffs[0] = coeffs[0]
    selected_coeffs[1] = coeffs[1]
    reconstructed = pywt.waverec(selected_coeffs, 'db4')

    if len(reconstructed) > len(signal_for_amplitude):
        reconstructed = reconstructed[:len(signal_for_amplitude)]

    for i in range(1, min_len):
        rr_interval = (r_peaks[i] - r_peaks[i - 1]) / fs
        if rr_interval > 0:
            hr = 60 / rr_interval
            if 30 <= hr <= 220:
                valid_hr = hr
            else:
               # print(f"⚠️ VYHODENÝ HR: {hr:.1f} bpm @ beat {i}")
                continue
        else:
            #print(f"⚠️ Chybný RR interval @ beat {i}")
            continue


        ro = r_onsets[i]
        rf = r_offsets[i]
        if ro is not None and rf is not None and rf > ro:
            qrs_rorr = (rf - ro) / fs * 1000
            if 40 <= qrs_rorr <= 180:
                valid_qrs = qrs_rorr
            if qrs_rorr < 40 or qrs_rorr > 180:
                #print(f"⚠️ VYHODENÝ QRS interval: {qrs_rorr:.1f} ms @ beat {i}")
                valid_qrs = np.nan
            # ⛔ Skip ak je QRS nerealistický alebo chýba
            if np.isnan(valid_qrs):
               # print(f"⏭️ Preskakujem beat {i} – žiadny spoľahlivý QRS interval")
                continue
        else:
            valid_qrs = np.nan

        r = r_peaks[i]
        search_start = r + int(0.15 * fs)
        search_end = r + int(0.4 * fs)
        if search_end >= len(reconstructed) or (i + 1 < len(r_peaks) and search_end > r_peaks[i + 1]):
           # print(f"⚠️ TWA_window mimo rozsah @ beat {i}")
            continue

        twa_wavelet = max(np.abs(reconstructed[search_start:search_end]))
       # print(f"🔍 Beat {i}: TWA_orig = {twa_orig:.3f} mV, TWA_wavelet = {twa_wavelet:.3f} mV")

        if not 0 < twa_wavelet <=1.0:
           # print(f"⚠️ Nevalidná TWA_wavelet @ beat {i}: {twa_wavelet:.3f} mV")
            continue

        # ⬇️ Ukladanie
        comparison_log.append({
            "Index": i,
            "HR": hr,
            "QRS_rorr": valid_qrs,
            "TWA": twa_wavelet,

        })

        features.append({
            "Index": int(i),
            "HR": float(hr),
            "QRS_interval": float(valid_qrs),
            "T_wave": float(twa_wavelet)
        })

        

        # Export
    df_cmp = pd.DataFrame(comparison_log)
    QRS_COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_cmp.to_csv(QRS_COMPARISON_CSV, index=False)
    print("✅ Porovnanie QRS + TWA uložené do results/reports/qrs_comparison.csv")
    return features
