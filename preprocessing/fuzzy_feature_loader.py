from config import MIT_DATA_PATH, FUZZY_FEATURE_CACHE, DATA_CACHE_DIR
import numpy as np
import pandas as pd
from preprocessing.load import load_mitbih_record
from preprocessing.filtering import lowpass_filter, dwt_filtering
from preprocessing.feature_extraction import extract_features_for_fuzzy
from preprocessing.annotation_mapping import map_annotations_to_peaks
from config import RECORD_NAMES

def load_or_extract_fuzzy_features():
    all_beat_counts_by_record = {}

    if FUZZY_FEATURE_CACHE.exists():
        print(f"📂 Načítavam features z cache: {FUZZY_FEATURE_CACHE}")
        data = np.load(FUZZY_FEATURE_CACHE, allow_pickle=True)
        return data["features"].tolist(), data["labels"].tolist(), all_beat_counts_by_record

    print("🛠️ Cache neexistuje – spúšťam extrakciu...")
    all_features, all_labels = [], []

    for record in RECORD_NAMES:
        try:
            signal, fs, r_peak_positions, beat_types, beat_counts, annotation = load_mitbih_record(
                record_name=record, path=str(MIT_DATA_PATH) + "/")
            all_beat_counts_by_record[record] = beat_counts

            filtered_signal = dwt_filtering(lowpass_filter(signal, fs))
            signal_for_amplitude = filtered_signal

            detected_r_peaks = r_peak_positions
            features = extract_features_for_fuzzy(signal_for_amplitude, fs, detected_r_peaks)
            detected_r_peaks = detected_r_peaks[1:]
            labels = map_annotations_to_peaks(detected_r_peaks, annotation.sample, annotation.symbol)

            for feat, label in zip(features, labels):
                if label != "Unknown":
                    all_features.append(feat)
                    all_labels.append(label)

            print(f"✅ Spracovaný záznam {record}: {len(features)} úderov")

        except Exception as e:
            print(f"⚠️ Chyba pri zázname {record}: {e}")

    # Uloženie do cache
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FUZZY_FEATURE_CACHE, features=np.array(all_features, dtype=object), labels=np.array(all_labels))
    print(f"💾 Dataset uložený do: {FUZZY_FEATURE_CACHE}")

    return all_features, all_labels, all_beat_counts_by_record