# scripts/optimize_twa.py
import itertools, json, numpy as np
from classifiers.fuzzy_classifier import FuzzyClassifier
from metrics import beat_level_metrics        # ← názov podľa teba
from main2 import load_or_compute_features
from config import ALL_RECORDS
import pandas as pd
grid = np.linspace(0.40, 0.60, 5)          # 0.40, 0.45, …, 0.60
best = {"macro_f1": -1}
feature_cache: dict[str, pd.DataFrame] = {}

def get_features(rec: str):
    if rec not in feature_cache:
        feature_cache[rec] = load_or_compute_features(rec)
    return feature_cache[rec]
clf = FuzzyClassifier()                    # jediná inštancia!

for low, high in itertools.product(grid, grid):
    if low >= high:            # zachovaj aspoň malý prekryv
        continue

    # 1) posuň MF pre T-vlnu
    clf.set_twa_mf(low, high)

    # 2) vyhodnoť subset záznamov (aby to bežalo rýchlo)
    yT, yP, cent = [], [], []
    for rec in ALL_RECORDS[:10]:
        feats = get_features(rec)
        t, p, c = clf.predict_batch(feats, top_n=0)
        yT.extend(t); yP.extend(p); cent.extend(c)

    m = beat_level_metrics(yT, yP, y_score=cent)
    if m["macro_f1"] > best["macro_f1"]:
        best = {"low": low, "high": high, **m}

print("Najlepšie nastavenie:")
print(json.dumps(best, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))