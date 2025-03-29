# preprocessing/load.py
import wfdb
from collections import Counter

EXCLUDED_ANNOTATIONS = {
    "+", "~", "|", "Q", "/", "f", "x", "\""
}



def load_mitbih_record(record_name, path="./data/mit/"):
    """
    Načíta EKG záznam z MIT-BIH databázy a vráti len validné údery na základe anotácií.

    Vrátené:
    - signal: 1D pole pôvodného EKG signálu (obsahuje aj nevalidné údery – nefiltruje sa)
    - fs: vzorkovacia frekvencia
    - filtered_r_peak_positions: pozície R-vĺn platných úderov
    - filtered_beat_types: typy týchto úderov (napr. 'N', 'V', 'A', ...)
    - beat_counts: počet výskytov jednotlivých platných typov úderov
    - annotation: pôvodné anotácie (kompletné, ak treba pre ďalšiu analýzu)
    """
    try:
        record = wfdb.rdrecord(f"{path}{record_name}")
        annotation = wfdb.rdann(f"{path}{record_name}", 'atr')
        fs = record.fs
        leads = record.sig_name
        print(record.units)

        # Načítanie signálu z prvého kanála
        if hasattr(record, "p_signal") and record.p_signal is not None:
            if record.p_signal.ndim == 2:
                signal = record.p_signal[:, 0]
            else:
                signal = record.p_signal
        else:
            raise ValueError("record.p_signal je None alebo neexistuje")

        # Filtrovanie len validných úderov
        all_r_peak_positions = annotation.sample
        all_beat_types = annotation.symbol

        filtered_r_peak_positions = []
        filtered_beat_types = []

        for r_pos, b_type in zip(all_r_peak_positions, all_beat_types):
            if b_type not in EXCLUDED_ANNOTATIONS:
                filtered_r_peak_positions.append(r_pos)
                filtered_beat_types.append(b_type)

        beat_counts = Counter(filtered_beat_types)

        return signal, fs, filtered_r_peak_positions, filtered_beat_types, beat_counts, annotation

    except Exception as e:
        print(f"❌ Error loading record {record_name}: {e}")
        return None, None, None, None, None

def summarize_loaded_beat_counts(all_beat_counts_by_record):
    """
    all_beat_counts_by_record: dict[str, Counter]
        kľúč = názov záznamu, hodnota = Counter validných beat typov
    """
    from collections import Counter

    total_counter = Counter()

    for record_name, counter in all_beat_counts_by_record.items():
        total_counter.update(counter)

    print("✅ Súhrn validných úderov zo všetkých záznamov:")
    for b_type, count in total_counter.items():
        print(f"🔹 Úder '{b_type}': {count} výskytov")

    return total_counter