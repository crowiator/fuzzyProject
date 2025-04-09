# preprocessing/fuzzy_features_loader.py
# Import vlastných funkcií pre spracovanie signálu a extrakciu príznakov
from config import MIT_DATA_PATH, FUZZY_FEATURE_CACHE, DATA_CACHE_DIR
import numpy as np
# Import vlastných funkcií pre spracovanie signálu a extrakciu príznakov
from preprocessing.load import load_mitbih_record
from preprocessing.filtering import lowpass_filter, dwt_filtering
from preprocessing.feature_extraction import extract_features_for_fuzzy
from preprocessing.annotation_mapping import map_annotations_to_peaks
from config import RECORD_NAMES

"""
Táto funkcia buď načíta dáta z cache, ak už boli predtým spracované a uložené,
alebo — ak cache neexistuje — spracuje všetky záznamy z MIT-BIH databázy.
Počas tohto spracovania aplikuje filtre na EKG signál,
extrahuje fuzzy-príznaky (features) z okolia R-vĺn 
a následne tieto údery mapuje na anotácie (typy srdcových úderov). 
Výsledné príznaky a labely sú potom uložené do cache súboru, 
čím sa výrazne zrýchli ich budúce použitie bez nutnosti opakovanej extrakcie.
"""


def load_or_extract_fuzzy_features():
    # Slovník na uloženie počtu úderov pre každý záznam
    all_beat_counts_by_record = {}

    # Ak cache existuje, načítame predspracované features a labely
    if FUZZY_FEATURE_CACHE.exists():
        print(f"Loading features from cache: {FUZZY_FEATURE_CACHE}")
        data = np.load(FUZZY_FEATURE_CACHE, allow_pickle=True)
        return data["features"].tolist(), data["labels"].tolist(), all_beat_counts_by_record

    # Ak cache neexistuje, spustíme extrakciu
    print("Cache does not exist – starting extraction...")
    all_features, all_labels = [], []

    # Spracovanie každého záznamu zo zoznamu RECORD_NAMES
    for record in RECORD_NAMES:
        try:
            # Načítanie EKG signálu, frekvencie vzorkovania, R-vĺn, anotácií a počtu úderov
            signal, fs, r_peak_positions, beat_types, beat_counts, annotation = load_mitbih_record(
                record_name=record, path=str(MIT_DATA_PATH) + "/")

            # Uloženie počtu úderov pre daný záznam
            all_beat_counts_by_record[record] = beat_counts

            # Filtrovanie signálu: lowpass + wavelet (DWT)
            filtered_signal = dwt_filtering(lowpass_filter(signal, fs))
            signal_for_amplitude = filtered_signal

            # Detegované R-vlny (od 1. pozície, nie od 0) a extrakica príznakov
            detected_r_peaks = r_peak_positions
            features = extract_features_for_fuzzy(signal_for_amplitude, fs, detected_r_peaks)
            detected_r_peaks = detected_r_peaks[1:]  # Posun, aby sedeli s features

            # Mapovanie anotácií (labelov) na detegované R-vlny
            labels = map_annotations_to_peaks(
                detected_r_peaks,
                annotation.sample,
                annotation.symbol)

            # Pridanie len známych labelov (bez "Unknown") do výstupu
            for feat, label in zip(features, labels):
                if label != "Unknown":
                    all_features.append(feat)
                    all_labels.append(label)

            print(f"Processed record {record}: {len(features)} beat")

        except Exception as e:
            # Ak nastane chyba, vypíšeme info o chybe, ale pokračujeme
            print(f"Error processing record  {record}: {e}")

    # Vytvorenie adresára pre cache, ak ešte neexistuje
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Uloženie extrahovaných dát do cache

    np.savez(FUZZY_FEATURE_CACHE, features=np.array(all_features, dtype=object), labels=np.array(all_labels))
    print(f"Dataset is saved into : {FUZZY_FEATURE_CACHE}")

    # Návrat všetkých príznakov, labelov a počtov úderov
    return all_features, all_labels, all_beat_counts_by_record
