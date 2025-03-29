# preprocessing/annotation_mapping.py
import numpy as np

ANNOTATION_TO_FUZZY = {
    "N": "Normal",     # Normálny sínusový úder

    # Moderate – bežné poruchy rytmu, bloky, extrasystoly
    "L": "Moderate",   # Blokáda ľavého Tawarovho ramienka (LBBB)
    "R": "Moderate",   # Blokáda pravého Tawarovho ramienka (RBBB)
    "A": "Moderate",   # Predčasný predsieňový úder (APB)
    "a": "Moderate",   # Aberantný predsieňový úder
    "J": "Moderate",   # Junkčný predčasný úder
    "S": "Moderate",   # Supraventrikulárny predčasný úder
    "F": "Moderate",   # Fúzovaný úder (normálny + PVC)
    "e": "Moderate",   # Predsieňový únikový úder
    "j": "Moderate",   # Junkčný únikový úder

    # Severe – potenciálne život ohrozujúce údery
    "V": "Severe",     # Predčasný komorový úder (PVC)
    "E": "Severe",     # Komorový únikový úder (idioventrikulárny)
    "!": "Severe",     # Vlna komorového fluttera (súčasť VF)
    "[": "Severe",     # Začiatok VF/flutter epizódy
    "]": "Severe",     # Koniec VF/flutter epizódy
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