# preprocessing/annotation_mapping.py
import numpy as np
# preprocessing/annotation_mapping.py
EXCLUDED_ANN = {"+", "[", "]", "~", "|", '"', "x", "U"}
ANNOTATION_TO_FUZZY = {
    "N": "normalna", ".": "normalna",

    # SVEB, artefakty, pacemaker → mierna
    "L": "mierna", "R": "mierna", "e": "mierna", "j": "mierna",
    "A": "mierna", "a": "mierna", "J": "mierna", "S": "mierna",
    "P": "mierna", "/": "mierna", "f": "mierna",

    # komorové, fúzia, asystólia → **zavazna**
    "V": "zavazna", "E": "zavazna", "!": "zavazna", "F": "zavazna",
}


def map_annotations_to_peaks(r_peaks, ann_samples, ann_symbols):
    """Ku každému R-vrcholu priradí najbližší MIT-BIH symbol a premapuje na normalna/mierna/zavazna."""
    mapped = []
    for r in r_peaks:
        idx = np.argmin(np.abs(ann_samples - r))
        symbol = ann_symbols[idx]
        if symbol not in EXCLUDED_ANN:               # ← tu bolo zle meno
            mapped.append(ANNOTATION_TO_FUZZY.get(symbol, "normalna"))
    return mapped