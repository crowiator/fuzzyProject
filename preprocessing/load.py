# preprocessing/load.py

import wfdb
from collections import Counter
from config import MIT_DATA_PATH, FUZZY_FEATURE_CACHE, DATA_CACHE_DIR, FUZZY_FEATURE_CACHE_CAMFEA, RECORD_NAMES
# Zoznam anotácií, ktoré nechceme zahrnúť do analýzy (napr. artefakty, chybné hodnoty)
EXCLUDED_ANNOTATIONS = {"+", "~", "|"}

"""
Táto funkcia načíta EKG signál a anotácie z MIT-BIH databázy pre daný záznam.
Zo všetkých anotovaných úderov vyfiltruje len tie, ktoré sú považované za "validné"
(t.j. nepatria medzi vylúčené symboly ako napr. "+", "~", "|", atď.).
 Vracia samotný signál, sampling rate, pozície R-vĺn a typy týchto úderov,
  ako aj ich štatistiku pomocou Counter
"""


def load_mitbih_record(record_name,  path=str(MIT_DATA_PATH) + "/"):
    """
    Načíta EKG záznam z MIT-BIH databázy a vráti len validné údery na základe anotácií.

    Vrátené:
    - signal: 1D pole pôvodného EKG signálu
    - fs: vzorkovacia frekvencia
    - filtered_r_peak_positions: pozície R-vĺn platných úderov
    - filtered_beat_types: typy týchto úderov (napr. 'N', 'V', 'A', ...)
    - beat_counts: počet výskytov jednotlivých platných typov úderov
    - annotation: pôvodné anotácie (kompletné, ak treba pre ďalšiu analýzu)
    """
    try:
        # Načítanie samotného EKG signálu a anotácií
        record = wfdb.rdrecord(f"{path}{record_name}")
        annotation = wfdb.rdann(f"{path}{record_name}", 'atr')
        fs = record.fs  # vzorkovacia frekvencia
        print(record.sig_name)
        # Načítanie signálu z prvého kanála (ak existuje a nie je None)
        if hasattr(record, "p_signal") and record.p_signal is not None:
            if record.p_signal.ndim == 2:
                signal = record.p_signal[:, 0]
            else:
                signal = record.p_signal
        else:
            raise ValueError("record.p_signal is None or not exist")

        # Filtrovanie len validných úderov
        all_r_peak_positions = annotation.sample
        all_beat_types = annotation.symbol
        filtered_r_peak_positions = []
        filtered_beat_types = []

        # Prejdi každé R-maximum a jeho typ, a ulož len validné
        for r_pos, b_type in zip(all_r_peak_positions, all_beat_types):
            if b_type not in EXCLUDED_ANNOTATIONS:
                filtered_r_peak_positions.append(r_pos)
                filtered_beat_types.append(b_type)

        # Spočítaj výskyt jednotlivých typov úderov
        beat_counts = Counter(filtered_beat_types)
        return signal, fs, filtered_r_peak_positions, filtered_beat_types, beat_counts, annotation

    # Chybové hlásenie, ak sa niečo nepodarí načítať
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        return None, None, None, None, None


def summarize_loaded_beat_counts(all_beat_counts_by_record):
    """
        Súhrnná štatistika výskytu jednotlivých typov úderov zo všetkých načítaných záznamov.

        Parametre:
        - all_beat_counts_by_record: dict[str, Counter]
            kľúč = názov záznamu, hodnota = Counter validných beat typov
        """

    total_counter = Counter()

    # Spojenie štatistík zo všetkých záznamov
    for record_name, counter in all_beat_counts_by_record.items():
        total_counter.update(counter)

    # Výpis štatistík do konzoly
    print("Summary of valid beats from all records:")
    for b_type, count in total_counter.items():
        print(f"Beat '{b_type}': {count} ocurrencies")

    return total_counter

 # Spracovanie každého záznamu zo zoznamu RECORD_NAMES
for record in RECORD_NAMES:
    try:
        # Načítanie EKG signálu, frekvencie vzorkovania, R-vĺn, anotácií a počtu úderov
        signal, fs, r_peak_positions, beat_types, beat_counts, annotation = load_mitbih_record(
                record_name=record, path=str(MIT_DATA_PATH) + "/")
    except Exception as e:
        # Ak nastane chyba, vypíšeme info o chybe, ale pokračujeme
        print(f"Error processing record  {record}: {e}")