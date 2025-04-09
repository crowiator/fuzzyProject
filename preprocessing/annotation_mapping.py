# preprocessing/annotation_mapping.py
import numpy as np

ANNOTATION_TO_FUZZY = {
    "N": "Normal",  # normálny sínusový úder

    "L": "Moderate",  # LBBB môže indikovať závažnejší stav
    "R": "Moderate",  # RBBB môže indikovať závažnejší stav
    "e": "Moderate",  # predsieňový únikový úder môže indikovať poruchu
    "j": "Moderate",  # junkčný únikový úder indikujúci poruchu vedenia
    "A": "Moderate",  # predsieňový extrasystol (riziko fibrilácie predsiení)
    "a": "Moderate",  # aberantný predsieňový extrasystol
    "J": "Moderate",  # junkčný extrasystol
    "S": "Moderate",  # supraventrikulárny extrasystol
    "F": "Moderate",  # fúzny úder naznačuje komorovú ektopickú aktivitu

    "V": "Severe",    # komorová extrasystola (potenciálne riziková)
    "E": "Severe",    # komorový únikový úder (potenciálne rizikový)
    "!": "Severe",    # flutter komôr (život ohrozujúci)
    "[": "Severe",    # začiatok fibrilácie/flutteru komôr (život ohrozujúci)
    "]": "Severe",    # koniec fibrilácie/flutteru komôr
}

def map_annotations_to_peaks(r_peaks, ann_samples, ann_symbols):
    """
    Priradí najbližšiu anotáciu ku každému detegovanému R-vrcholu
    a mapuje ju na fuzzy triedu (Normal / Moderate / Severe).
    """
    mapped_labels = []
    for r in r_peaks:
        idx = np.argmin(np.abs(ann_samples - r))
        symbol = ann_symbols[idx]
        fuzzy_label = ANNOTATION_TO_FUZZY.get(symbol, "Unknown")
        mapped_labels.append(fuzzy_label)
    return mapped_labels