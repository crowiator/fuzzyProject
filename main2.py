# main2.py
# =========
# =========================================================================
# Pipeline:
#   1. stiahne (alebo otvorí lokálne) MIT-BIH záznam
#   2. vyčistí signál a vyťaží príznaky (HR, QRS, T_amp)
#   3. na každý beat použije fuzzy klasifikátor (Mamdani)
#   4. porovná s „pravým štítkom“ z *.atr* a spočítá metriky
#   5. uloží CSV + vykreslí prehľadné grafy pre lekára
# =========================================================================
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from classifiers.traditional_models import (
        train_knn, train_decision_tree, train_random_forest, train_svm)
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from ecg_pro import load_record, clean_ecg, extract_features
from classifiers.fuzzy_classifier import FuzzyClassifier
from preprocessing.annotation_mapping import ANNOTATION_TO_FUZZY, EXCLUDED_ANN  # slovník
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import matplotlib.pyplot as plt
import matplotlib
from config import ALL_RECORDS, LEAD, OUT_DIR, FEAT_DIR, MF_CFG_PATH

matplotlib.use('TkAgg')
from metrics import beat_level_metrics, patient_metrics

FEAT_COLS_BASE = ["HR_bpm", "QRS_ms", "T_amp"]
FEAT_COLS_FUZZY = [
    "centroid_fuzzy",
    "mu_sum_normalna", "mu_sum_mierna", "mu_sum_zavazna"
]
# po vyrátaní predikcií …

# -------------------------------------------------------------------------
# 1) Jednorazové vytvorenie výstupných priečinkov
# -------------------------------------------------------------------------

RECORD_ID = "100"
FEAT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# 2) Extrakcia príznakov (HR, QRS, T_amp) na úrovni jedného beatu
# ──────────────────────────────────────────────────────────────────────────


def extract_features_manage(clean_signal, fs, ann, record_id):
    features_df = extract_features(clean_signal, fs, ann, record_id=record_id)
    n_total = len(features_df)
    # ⚠️ NOVÝ RIADOK – zahodí beaty s neželanou anotáciou
    features_df = features_df[~features_df["Annotation"].isin(EXCLUDED_ANN)].reset_index(drop=True)
    #  jednoduchá validácia/čistenie príznakov
    mask = (
            (features_df["QRS_ms"].between(30, 500) | features_df["QRS_ms"].isna()) &
            (features_df["T_amp"].notna())  # T_amp NaN môžete nahradiť 0
    )
    # HR nechajme len ako doplnkovú informáciu
    features_df.loc[~features_df["HR_bpm"].between(20, 220), "HR_bpm"] = np.nan
    features_df = features_df.loc[mask].reset_index(drop=True)
    features_df["bad_qrs"] = ~features_df["QRS_ms"].between(30, 500)
    features_df["bad_hr"] = ~features_df["HR_bpm"].between(20, 220)
    features_df["bad_t"] = features_df["T_amp"].isna()

    print(features_df[["bad_qrs", "bad_hr", "bad_t"]].mean() * 100)
    # ❹ spočítaj, koľko zostalo a koľko sa vyhodilo
    n_kept = len(features_df)
    n_dropped = n_total - n_kept
    pct_drop = n_dropped / n_total * 100

    print(f"Beats celkom : {n_total}")
    print(f"Zachovaných  : {n_kept}")
    print(f"Vyhodených   : {n_dropped}  ({pct_drop:.1f} %)")
    print(f"[{record_id}]  zachovaných {len(features_df)}/{n_total}  "
          f"({100 * len(features_df) / n_total:.1f} %)")
    n_after_excl = len(features_df)
    print(f"Odfiltrované (excluded ann): {n_total - n_after_excl}")
    return features_df


def clean_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
            (df["QRS_ms"].between(30, 500) | df["QRS_ms"].isna()) &
            (df["T_amp"].notna())
    )
    df.loc[~df["HR_bpm"].between(20, 220), "HR_bpm"] = np.nan
    df = df.loc[mask].reset_index(drop=True)
    return df


def load_or_compute_features(record_id: str):
    """
    Vracia: features_df, fs, clean_signal, ann
    - ak existuje ecg_<ID>_features_full.csv → len sa načíta,
      inak sa celé spracovanie vykoná a CSV sa uloží.
    """
    feat_path = FEAT_DIR / f"ecg_{record_id}_features_full.csv"
    if feat_path.is_file():
        #print(f"🔄  Načítavam uložené príznaky z {feat_path}")
        df = pd.read_csv(feat_path)
        # ⬇️ PRIDAJ TOTO: Explicitná filtrácia znova pri načítaní!
        df = df[~df["Annotation"].isin(EXCLUDED_ANN)].reset_index(drop=True)
        return clean_feature_df(df)

    # ak CSV neexistuje – vypočítaj
    raw, fs, ann = load_record(record_id, lead=LEAD)
    sig_df, _ = clean_ecg(raw, fs, method="custom")
    clean_sig = sig_df["ECG_Clean"].to_numpy()
    df = extract_features_manage(clean_sig, fs, ann, record_id)
    return df


# -------------------------------------------------------------------------
# 3) Fuzzy klasifikácia jedného DataFrame-u
# -------------------------------------------------------------------------

def run_fuzzy(clf, features_df):
    """
        Pre každý beat z features_df vypočíta fuzzy centroid, slovný label
        a uloží aj top 3 najaktivovanejšie pravidlá.
    """
    pred_centroid = []
    pred_label = []
    active_rules = []
    for _, row in features_df.iterrows():
        res = clf.predict(row["HR_bpm"],
                          row["QRS_ms"],
                          (row["T_amp"]),  # TWA je absolútna hodnota # toto pozriet
                          top_n=3)
        pred_centroid.append(res["centroid"])
        pred_label.append(res["label"].lower())  # → 'normalna' | 'mierna' | 'zavazna'
        active_rules.append(res["rules"])  # voliteľné – zoznam dictov

    # ----------------------------- 4) pravé triedy --------------------------

    true_label = [ANNOTATION_TO_FUZZY.get(sym, "unknown").lower()
                  for sym in features_df["Annotation"]]
    # ───────────────────────────
    #  Označenie invalid úderov
    # ───────────────────────────
    invalid_mask = [lbl == "invalid" for lbl in pred_label]
    n_invalid = sum(invalid_mask)
    if n_invalid:
        pct = n_invalid / len(pred_label) * 100
        print(f"⚠️  invalid beats: {n_invalid}/{len(pred_label)} "
              f"({pct:.1f} %) – zapíšem ich, ale nebudú v metrikách")

    # stĺpec Invalid do pôvodného DataFrame,
    # aby sa dostal až do finálneho CSV
    features_df["Invalid"] = invalid_mask

    return true_label, pred_centroid, pred_label, active_rules


# -------------------------------------------------------------------------
# Pomocná funkcia – uloží DataFrame s fuzzy výstupom
# -------------------------------------------------------------------------
def save_fuzzy_csv(df: pd.DataFrame, record_id: str) -> Path:
    """
    Uloží *df* do priečinka OUT_DIR s jednotným názvom
    a zároveň ho vráti (Path), aby sa dal prípadne použiť inde.
    """
    path = OUT_DIR / f"mitdb_{record_id}_fuzzy_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Výsledok uložený do {path}")
    return path

# -------------------------------------------------------------------------
# 4) Uloženie + klinický súhrn
# -------------------------------------------------------------------------
def get_final_report(true_label, pred_centroid, pred_label, active_rules, out, record_id):
    """
        Vytvorí finálny DataFrame + uloží CSV + vytlačí súhrn.
    """
    out = out
    out["TrueLabel"] = true_label
    out["FuzzyLabel"] = pred_label
    out["Centroid"] = pred_centroid
    out["Rules"] = active_rules  # JSON-ovateľné pole
    # po vašom out = features_df.copy() …
    out["Confidence"] = out["Centroid"].apply(
        lambda x: "💚" if x < 0.25 else ("🟠" if x < 0.6 else "🔴"))

    cols = ["R_index", "HR_bpm", "QRS_ms", "T_amp",
            "FuzzyLabel", "Centroid", "Confidence", "TrueLabel", "Rules"]
    report_df = out[cols]
    output_path = OUT_DIR / f"mitdb_{record_id}_fuzzy_results.csv"
    out.to_csv(output_path, index=False)
    print(f"✅ Výsledok uložený do {output_path}")
    # ----------------------------- klinický súhrn -----------------------------
    summary = (
        out["FuzzyLabel"]
        .value_counts(normalize=True)
        .reindex(["normalna", "mierna", "zavazna"], fill_value=0)
        .mul(100)
        .round(1)
    )

    risk_flag = (
        "✅ bez klinicky významnej arytmie"
        if summary["zavazna"] < 1
        else ("🟠 mierna arytmia" if summary["zavazna"] < 5 else "🔴 významná arytmia")
    )
    print("\n=== Súhrn ===")
    print(summary.to_frame("% úderov"))
    print("\nCelkové riziko:", risk_flag)
    return out


# 3) Zobrazenie závažných úderov priamo na EKG krivke


def show_severe_beat_ecg(out, fs, clean_signal):
    sev_idx = out.index[out["FuzzyLabel"] == "zavazna"]
    win = int(0.6 * fs)  # 0.6‑sekundové okno okolo úderu

    for i in sev_idx[:3]:  # ukáž maximálne prvé tri
        r_pos = int(out.loc[i, "R_index"])
        start = max(r_pos - win, 0)
        stop = min(r_pos + win, len(clean_signal) - 1)

        t_local = np.arange(start, stop) / fs
        plt.figure(figsize=(6, 2))
        plt.plot(t_local, clean_signal[start:stop], "k")
        plt.axvline(r_pos / fs, color="#d62728", lw=1.5, label="závažný úder")
        plt.title(f"Závažný úder #{i}  –  centroid={out.loc[i, 'Centroid']:.2f}")
        plt.xlabel("čas [s]")
        plt.ylabel("mV")
        plt.legend()
        plt.tight_layout()
        plt.show()


# ----------------------------- TOP pravidlá -------------------------------

# -------------------------------------------------------------------------
# 5) Pomocne funkcie na metriky a vizualizácie
# -------------------------------------------------------------------------
def get_top_rules(out):
    rules_long = (
        out["Rules"]
        .explode()
        .dropna()
        .apply(lambda d: d.get("rule") if isinstance(d, dict) else None)
        .value_counts()
        .head(10)
        .to_frame("count")
    )
    rules_long["%"] = (rules_long["count"] / len(out) * 100).round(2)

    print("\n=== TOP pravidlá (najčastejšie aktivované) ===")
    print(rules_long)


def get_confussion_matrix(out, labels):
    mask = out["TrueLabel"].isin(labels) & out["FuzzyLabel"].isin(labels)
    cm = confusion_matrix(out.loc[mask, "TrueLabel"],
                          out.loc[mask, "FuzzyLabel"],
                          labels=labels)
    print("\n=== Confusion matrix ===")
    print(pd.DataFrame(cm, index=[f"T_{l}" for l in labels],
                       columns=[f"P_{l}" for l in labels]))


def get_classification_report(out: pd.DataFrame, labels: list[str]) -> None:
    """
    Vypíše classification report len pre tie triedy, ktoré
    sa v danom zázname skutočne vyskytujú (v y_true).
    Varovanie UndefinedMetricWarning potlačíme – ak sa
    predsa len objaví ďalšie delenie nulou, nastaví sa 0.
    """
    # 1) vyfiltruj riadky, kde máme definované obidva štítky
    mask = out["TrueLabel"].isin(labels) & out["FuzzyLabel"].isin(labels)
    y_true = out.loc[mask, "TrueLabel"]
    y_pred = out.loc[mask, "FuzzyLabel"]

    # 2) ponechaj len triedy, ktoré sa NAOZAJ vyskytujú v y_true
    labels_present = [lab for lab in labels if (y_true == lab).any()]

    print("\n=== Classification report ===")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        print(
            classification_report(
                y_true,
                y_pred,
                labels=labels_present,
                digits=3,
                zero_division=0
            )
        )


FEAT_COLS = FEAT_COLS_BASE   # ⬅ sem si vieš pridať ďalšie príznaky


def _prepare_xy(df: pd.DataFrame,  *, feat_cols):
    X = df[feat_cols].to_numpy()
    y = df["TrueLabel"].to_numpy()
    # MIT-BIH: každý záznam ≈ jeden pacient → stĺpec 'Record'
    groups = df["Record"].to_numpy()

    # NaN → mediána stĺpcov (aby sa nevypustili celé riadky)
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)

    # Škálovanie (pomáha hlavne KNN a SVM)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, groups


def train_and_evaluate_traditional(df_all_beats: pd.DataFrame) -> dict:
    X, y, groups = _prepare_xy(df_all_beats, feat_cols=FEAT_COLS_BASE)

    # Jednorazový 80/20 split po pacientoch
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    models = {
        "KNN": train_knn(
            X_train, y_train, metric="euclidean",
            n_neighbors=7, weights="distance"
        ),
        "DT":  train_decision_tree(
            X_train, y_train, criterion="gini", max_depth=None,
            max_features=None, min_samples_leaf=1, min_samples_split=2
        ),
        "RF":  train_random_forest(
            X_train, y_train, criterion="gini", max_depth=None,
            max_features="sqrt", min_samples_leaf=1,
            min_samples_split=2, n_estimators=200
        ),
        "SVM": train_svm(
            X_train, y_train, C=2.0, degree=3,
            gamma="scale", kernel="rbf"
        ),
    }

    results = {}
    for name, mdl in models.items():
        y_pred = mdl.predict(X_test)

        # Skóre pre „závažnú“ – ak model podporuje pravdepodobnosti / decision_function
        y_score = None
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(X_test)
            if proba.ndim == 1 or proba.shape[1] == 1:
                y_score = proba.ravel()
            elif "zavazna" in mdl.classes_:
                idx = list(mdl.classes_).index("zavazna")
                y_score = proba[:, idx]
        elif hasattr(mdl, "decision_function"):
            decf = mdl.decision_function(X_test)
            if decf.ndim == 1:
                y_score = decf
            elif "zavazna" in mdl.classes_:
                idx = list(mdl.classes_).index("zavazna")
                y_score = decf[:, idx]

        if y_score is not None and y_score.ndim > 1:
            y_score = y_score.max(axis=1)

        results[name] = beat_level_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score
        )

    return results

# 2) Funkcia – len drobné premenovanie + FEAT_COLS už platí globálne
def train_and_evaluate_hybrid(df_all):
    X, y, groups = _prepare_xy(
        df_all,
        feat_cols=FEAT_COLS_BASE + FEAT_COLS_FUZZY
    )
    ...  # ← df_all, nie df_all_beats

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(splitter.split(X, y, groups))

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    models = {
        "KNN-hyb": train_knn(X_tr, y_tr, metric="euclidean",
                             n_neighbors=7, weights="distance"),
        "RF-hyb":  train_random_forest(X_tr, y_tr, criterion="gini",
                                       max_depth=None, max_features="sqrt",
                                       min_samples_leaf=1, min_samples_split=2,
                                       n_estimators=200),
        # pridaj ďalšie, keď chceš
    }

    results = {}
    for name, mdl in models.items():
        y_pred  = mdl.predict(X_te)
        y_score = None
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(X_te)
            if "zavazna" in mdl.classes_:
                y_score = proba[:, list(mdl.classes_).index("zavazna")]

        results[name] = beat_level_metrics(
            y_true=y_te,
            y_pred=y_pred,
            y_score=y_score
        )
    return results

def patient_level_metrics(df: pd.DataFrame,
                          *,
                          patient_col: str = "Record",
                          true_col: str = "TrueLabel",
                          pred_col: str = "FuzzyLabel",
                          index_col: str = "R_index",
                          fs: int = 360,
                          abnormal_labels: set[str] | None = None
                          ) -> dict[str, float]:
    """
    Pohodlnejší wrapper nad `patient_metrics`:
    vezme DataFrame a vráti slovník s metrikami.
    """
    if abnormal_labels is None:
        abnormal_labels = {"zavazna"}

    y_true = df[true_col].to_numpy()
    y_pred = df[pred_col].to_numpy()
    patient_ids = df[patient_col].to_numpy()
    timestamps = df[index_col].to_numpy() / fs

    sens, fa = patient_metrics(
        y_true, y_pred, patient_ids, timestamps, abnormal_labels
    )
    return {
        "patient_sensitivity": sens,
        "false_alarms_per_hour": fa,
    }
# -------------------------------------------------------------------------
# 6) Hlavný beh
# -------------------------------------------------------------------------
def main(mode: str = "fuzzy"):
    """
    mode = "fuzzy"  → spracuje (alebo načíta) výsledky fuzzy klasifikátora
    mode = "trad"   → preskočí fuzzy a vytrénuje/ohodnotí len tradičné modely
    """
    if mode not in {"fuzzy", "trad", "hybrid"}:
        raise ValueError("mode musí byť 'fuzzy' alebo 'trad'")

    clf = FuzzyClassifier()

    all_out, y_true_all, y_pred_all, cent_all = [], [], [], []

    # ---------- ①  FUZZY časť (len ak treba) -------------------------------
    if mode == "fuzzy":
        for rec in ALL_RECORDS:
            csv_path = OUT_DIR / f"mitdb_{rec}_fuzzy_results.csv"

            if csv_path.is_file():
                print(f"📄 {rec} – načítavam existujúci CSV …")
                df = pd.read_csv(csv_path)

                # ak historický CSV ešte nemal stĺpec Invalid, doplň ho
                if "Invalid" not in df.columns:
                    df["Invalid"] = df["FuzzyLabel"] == "invalid"

                y_true = df["TrueLabel"].tolist()
                y_pred = df["FuzzyLabel"].tolist()
                centroid = df["Centroid"].tolist()

            else:
                feats = load_or_compute_features(rec)
                y_true, centroid, y_pred, rules = run_fuzzy(clf, feats)

                df = feats.copy()
                df["TrueLabel"] = y_true
                df["FuzzyLabel"] = y_pred
                df["Centroid"] = centroid
                df["Rules"] = rules
                df["Invalid"] = df["FuzzyLabel"] == "invalid"

                # ulož, aby nabudúce už existoval
                save_fuzzy_csv(df, rec)

            df["Record"] = rec
            all_out.append(df)
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
            cent_all.extend(centroid)

        df_all = pd.concat(all_out, ignore_index=True)

        # ──────────────────────────────────────────────────────
        #   METRIKY len pre validné údery (Invalid == False)
        # ──────────────────────────────────────────────────────
        valid_df = df_all[df_all["FuzzyLabel"] != "invalid"]

        print("\n=== Beat-level metriky (validné údery) ===")
        metrics = beat_level_metrics(
            valid_df["TrueLabel"],
            valid_df["FuzzyLabel"],
            y_score=valid_df["Centroid"],
            keep_unknown=False
        )
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k:>25}: {v:.3f}")
            elif k == "confusion_matrix":
                print(f"{k:>25}:\n{v}")

        print("\n==== FUZZY – PATIENT-LEVEL ====")
        for k, v in patient_level_metrics(valid_df).items():
            print(f"{k:25}: {v:.3f}")
    # ---------- ②  TRADIČNÉ modely (len ak treba) -------------------------
    if mode == "trad":
        # ak fuzzy CSV ešte nemáte v pamäti, načítajte ich
        if not all_out:
            for rec in ALL_RECORDS:
                p = OUT_DIR / f"mitdb_{rec}_fuzzy_results.csv"
                if p.is_file():
                    df = pd.read_csv(p)

                    # ak historické CSV nemá stĺpec Invalid, doplň ho
                    if "Invalid" not in df.columns:
                        df["Invalid"] = df["FuzzyLabel"] == "invalid"

                    all_out.append(df)

        if not all_out:
            raise RuntimeError("Nemám dáta s TrueLabel – najprv spusť fuzzy mód.")

        df_all = pd.concat(all_out, ignore_index=True)

        # ───────────────────────────────────────
        #  Validné údery = Invalid == False
        # ───────────────────────────────────────
        df_valid = df_all[~df_all["Invalid"]].reset_index(drop=True)

        # natrénuj a ohodnoť tradičné modely
        trad_res = train_and_evaluate_traditional(df_valid)

        print("\n================ TRADIČNÉ MODELY (validné údery) ================\n")
        for mdl, metr in trad_res.items():
            print(f"---- {mdl} ----")
            for k, v in metr.items():
                txt = f"{v:.4f}" if np.isscalar(v) else v
                print(f"{k:20s}: {txt}")
            print()
    if mode == "hybrid":

        # 1) načítaj CSV-ka
        df_train = pd.read_csv("datasets/train_fuzzy.csv")
        df_val = pd.read_csv("datasets/val_fuzzy.csv")

        # 2) zmeň ground-truth z MIT symbolov → 3 klinické triedy
        for df in (df_train, df_val):
            df["TrueLabel"] = (
                df["Annotation"]
                .map(ANNOTATION_TO_FUZZY)  # slovník {'N':'normalna', …}
                .str.lower()
            )
            df.dropna(subset=["TrueLabel"], inplace=True)  # vyhoď unknown
            df = df[df["TrueLabel"] != "unknown"]

        # 3) spoj tréning + val
        df_all = pd.concat([df_train, df_val], ignore_index=True)

        # 4) natrénuj a ohodnoť hybrid
        hybr_res = train_and_evaluate_hybrid(df_all)

        # 5) pekný výpis
        print("\n================ HYBRIDNÉ MODELY ================\n")
        for mdl, metr in hybr_res.items():
            print(f"---- {mdl} ----")
            for k, v in metr.items():
                txt = f"{v:.4f}" if np.isscalar(v) else v
                print(f"{k:20s}: {txt}")
            print()

# -------------------------------------------------------------------------
if __name__ == "__main__":

    main(mode="trad")