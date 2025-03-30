from pathlib import Path
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from preprocessing.segmentation import segment_heartbeats
from preprocessing.annotation_mapping import map_annotations_to_peaks
from preprocessing.feature_extraction import extract_features_for_fuzzy
from preprocessing.load import load_mitbih_record
from config import RECORD_NAMES
def prepare_shared_cnn_dataset(pre_R=0.2, post_R=0.4):
    BASE_DIR = Path(__file__).resolve().parents[1]  # hlavný priečinok projektu
    cache_path = BASE_DIR / "results" / "data" / "cnn_segments_fuzzy.npz"
    reports_dir = BASE_DIR / "results" / "reports"

    # Vytvor adresáre ak neexistujú
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Načítanie z cache
    if cache_path.exists():
        print(f"📂 Načítavam dataset zo súboru: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X_segments"], data["X_fuzzy"], data["y_encoded"], data["label_classes"]

    # Vytvorenie dát
    X_segments, X_fuzzy, y_labels = [], [], []

    for record in RECORD_NAMES:
        try:
            signal, fs, r_peaks, _, _, annotation = load_mitbih_record(record, path="../data/mit/")
            segments = segment_heartbeats(signal, r_peaks, fs, pre_R=pre_R, post_R=post_R)
            adjusted_peaks = r_peaks[1:len(segments)+1]
            fuzzy_features = extract_features_for_fuzzy(signal, fs, adjusted_peaks)
            labels = map_annotations_to_peaks(adjusted_peaks, annotation.sample, annotation.symbol)

            for seg, feat, label in zip(segments, fuzzy_features, labels):
                if label != "Unknown":
                    hr = feat["HR"]
                    qrs = feat["QRS_interval"]
                    twa = feat["T_wave"]
                    score = feat.get("Fuzzy_Score", 0.0)
                    if not any(np.isnan([hr, qrs, twa])):
                        X_segments.append(seg)
                        X_fuzzy.append([hr, qrs, twa, score])
                        y_labels.append(label)

            print(f"✅ Záznam {record} pripravený: {len(X_segments)} platných segmentov")
        except Exception as e:
            print(f"⚠️ Chyba pri spracovaní {record}: {e}")

    # Konverzia a uloženie
    X_segments = np.array(X_segments)
    X_fuzzy = np.array(X_fuzzy)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_labels)

    np.savez(cache_path,
             X_segments=X_segments,
             X_fuzzy=X_fuzzy,
             y_encoded=y_encoded,
             label_classes=encoder.classes_)
    print(f"💾 Dataset uložený do: {cache_path}")
    print("📦 Dataset pripravený:", X_segments.shape[0], "záznamov")

    return X_segments, X_fuzzy, y_encoded, encoder