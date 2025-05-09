"""
add_fuzzy_outputs.py
--------------------
Načíta existujúcu tabuľku príznakov, pre každý úder zavolá
FuzzyClassifier.predict() a pripojí stĺpce:

    centroid_fuzzy   – 0-1 skóre
    mu_normalna
    mu_mierna
    mu_zavazna
    label_fuzzy      – WTA trieda

Výsledok sa uloží do  <pôvodný_názov>_fuzzy.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from classifiers.fuzzy_classifier import FuzzyClassifier
from config import OUT_TRAIN_FUZZY
from preprocessing.annotation_mapping import ANNOTATION_TO_FUZZY


# 1 – zisti koreň projektu (riadok pridaj hneď za importy)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2 – nastav priečinky
SRC_FUZZY = PROJECT_ROOT / "exported_features_fuzzy"
SRC_RAW   = PROJECT_ROOT / "exported_features"
DST_DIR   = PROJECT_ROOT / "datasets"
OUT_TRAIN_FUZZY = SRC_FUZZY              # ak chceš používať tú istú premennú

DST_DIR.mkdir(exist_ok=True)
SRC_FUZZY.mkdir(exist_ok=True)
INPUT_CSV = Path("features_train.csv")      # ← zmeň ak treba

OUT_TRAIN_FUZZY.mkdir(exist_ok=True)
OUTPUT_CSV = INPUT_CSV.with_name(INPUT_CSV.stem + "_fuzzy.csv")
TRAIN_RECORDS = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
         "111", "112", "113", "114", "115", "116", "117", "118", "119",
         "121", "122", "123", "124",
         "201", "202", "203", "205", "207", "212"]
VAL_RECORDS = ["208", "209", "210", "213", "214", "215", "217", "219", "220"]
RECORDS = TRAIN_RECORDS + VAL_RECORDS
TEST = ["200","221","222","223","228","230","231","232","233","234"]
fuzzy = FuzzyClassifier()
def load_test_val_records():
    for rec in RECORDS:
        input_csv  = Path(f"../exported_features/ecg_{rec}_features_full.csv")
        output_csv = OUT_TRAIN_FUZZY / f"ecg_{rec}_features_full_fuzzy.csv"

        print("Načítavam:", input_csv)
        df = pd.read_csv(input_csv)

        # (listy na zbieranie hodnôt)
        centroids, mu_norm, mu_mid, mu_sev, lbl_f = [], [], [], [], []
        mu_sum_norms, mu_sum_mierns, mu_sum_zavs = [], [], []

        for hr, qrs, twa in df[["HR_bpm", "QRS_ms", "T_amp"]].itertuples(index=False):
            res = fuzzy.predict(hr, qrs, twa, top_n=0)
            # ——— ① ošetri „Invalid“ výstup ——————————————
            if not isinstance(res, dict):  # tuple → invalid
                centroids.append(np.nan)
                mu_norm.append(0.0)
                mu_mid.append(0.0)
                mu_sev.append(0.0)
                lbl_f.append("invalid")
                mu_sum_norms.append(0.0)
                mu_sum_mierns.append(0.0)
                mu_sum_zavs.append(0.0)
                continue  # preskoč zvyšok iterácie
            # ———————————————
            centroids.append(res["centroid"])
            mu_norm.append(res["memberships"]["normalna"])
            mu_mid.append(res["memberships"]["mierna"])
            mu_sev.append(res["memberships"]["zavazna"])
            lbl_f.append(res["label"])

            mu_sum_norms.append(
                sum(r["μ"] for r in res["rules"] if r["rule"].endswith("normalna")))
            mu_sum_mierns.append(
                sum(r["μ"] for r in res["rules"] if r["rule"].endswith("mierna")))
            mu_sum_zavs.append(
                sum(r["μ"] for r in res["rules"] if r["rule"].endswith("zavazna")))

        # priradenie stĺpcov
        df["centroid_fuzzy"]  = centroids
        df["mu_normalna"]     = mu_norm
        df["mu_mierna"]       = mu_mid
        df["mu_zavazna"]      = mu_sev
        df["label_fuzzy"]     = lbl_f
        df["mu_sum_normalna"] = mu_sum_norms
        df["mu_sum_mierna"]   = mu_sum_mierns
        df["mu_sum_zavazna"]  = mu_sum_zavs
        df["TrueLabel"] = (
            df["Annotation"]
            .map(ANNOTATION_TO_FUZZY)  # mapuj každú anotáciu
            .str.lower()  # 'Normalna' → 'normalna'
            .fillna("unknown")  # ak nie je v mape
        )
        df.to_csv(output_csv, index=False)
        print("✅ uložené →", output_csv)

def concat(ids, src, outfile):
    dfs = []
    for rec in ids:
        f = src / f"ecg_{rec}_features_full{'_fuzzy' if src is SRC_FUZZY else ''}.csv"
        if not f.exists():
            print("⚠️  chýba", f); continue
        df = pd.read_csv(f)

        # ------ vyhoď 'unknown' a nechaj len tri triedy ----------
        df = df[df["TrueLabel"].isin(["normalna", "mierna", "zavazna"])]

        dfs.append(df)

    out_path = DST_DIR / outfile
    pd.concat(dfs, ignore_index=True).to_csv(out_path, index=False)
    print(f"✅ uložené → {out_path}  ({sum(len(d) for d in dfs):,} riadkov)")

def contact_together():
    concat(TRAIN_RECORDS, SRC_FUZZY, "train_fuzzy.csv")
    concat(VAL_RECORDS, SRC_FUZZY, "val_fuzzy.csv")
    concat(TEST,  SRC_RAW,   "test_raw.csv")      # test zatiaľ bez fuzzy stĺpcov


if __name__ == "__main__":
    print("SRC_FUZZY →", SRC_FUZZY)
    print("  existuje:", SRC_FUZZY.exists())
   # load_test_val_records()    # ① vypočíta fuzzy stĺpce pre train+val
    contact_together()