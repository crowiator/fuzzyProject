# tune_mf_percentiles.py
import numpy as np, pandas as pd, glob, json
from pathlib import Path
from classifiers.fuzzy_classifier import FuzzyClassifier
import skfuzzy as fuzz

# 1) Načítaj všetky súbory a vyfiltruj len validné labely
def load_dataset(folder="mitdb_fuzzy_results"):
    dfs = [pd.read_csv(f) for f in glob.glob(f"{folder}/*.csv")]
    df  = pd.concat(dfs, ignore_index=True)
    df  = df[df["TrueLabel"].isin(["normalna","mierna","zavazna"])]
    df  = df.rename(columns={"HR_bpm":"HR","QRS_ms":"QRS","T_amp":"T"})
    return df[["HR","QRS","T"]].astype(float)

df = load_dataset()

# 2) Vypočítaj percentily (prispôsob si, koľko MF chceš)
P = df.quantile([0.1, 0.5, 0.9])
hr_lo, hr_mid, hr_hi = P.loc[0.1,"HR"], P.loc[0.5,"HR"], P.loc[0.9,"HR"]
qrs_lo, qrs_mid, qrs_hi = P.loc[0.1,"QRS"], P.loc[0.5,"QRS"], P.loc[0.9,"QRS"]
t_lo, t_mid, t_hi   = P.loc[0.1,"T"],   P.loc[0.5,"T"],   P.loc[0.9,"T"]

clf = FuzzyClassifier()              # -- už existujúci rule-base

# 3) Prepíš uzly „in-place“ (bez deepcopy!)
def reset_triangular(term, a,b,c): term.mf = fuzz.trimf(term.parent.universe,[a,b,c])

reset_triangular(clf.hr["nizka"],     20,                    (20+hr_lo)/2, hr_mid)
reset_triangular(clf.hr["normalna"],  hr_lo,                 hr_mid,       hr_hi)
reset_triangular(clf.hr["vysoka"],    hr_mid,                (hr_mid+hr_hi)/2, 220)

reset_triangular(clf.qrs["kratky"],   40,                    (40+qrs_lo)/2, qrs_mid)
reset_triangular(clf.qrs["normalny"], qrs_lo,                qrs_mid,      qrs_hi)
reset_triangular(clf.qrs["dlhy"],     qrs_mid,               (qrs_mid+qrs_hi)/2, 400)

reset_triangular(clf.twa["nizka"],    df["T"].min(),         (df["T"].min()+t_lo)/2, t_mid)
reset_triangular(clf.twa["normalna"], t_lo,                  t_mid,        t_hi)
reset_triangular(clf.twa["vysoka"],   t_mid,                 (t_mid+t_hi)/2, df["T"].max())

# 4) Nastav benevolentnejší threshold pre unknown
clf.UNKNOWN_THRESHOLD = 0.08   # pridaj atribút / alebo zmeň v predict()

# 5) Otestuj na validačnom sete
from sklearn.metrics import f1_score
label_map = {"normalna":0,"mierna":1,"zavazna":2}
y_true = df.index.map(lambda i: label_map[df.iloc[i//3]["TrueLabel"]])  # skratený príklad
y_pred = [ label_map.get(clf.predict(*row)["label"], 3) for _,row in df.iterrows() ]
print("macro-F1 =", f1_score(y_true,y_pred,labels=[0,1,2],average="macro"))