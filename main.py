import matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
matplotlib.use("TkAgg")
from imblearn.over_sampling import SMOTE
from preprocessing.load import load_mitbih_record, summarize_loaded_beat_counts
from preprocessing.segmentation import segment_heartbeats
from classifiers.traditional_models import train_decision_tree, train_random_forest, train_svm, evaluate_model
from classifiers.fuzzy_classifier import predict_arrhythmia
from utils.visualization import plot_signal_comparison
from config import MIT_DATA_PATH, FUZZY_FEATURE_CACHE, DATA_CACHE_DIR, REPORTS_DIR
from preprocessing.fuzzy_feature_loader import load_or_extract_fuzzy_features
from train.train_cnn_models import run_cnn_models
from crossval import cross_validate_model, cross_validate_cnn_models
from classifiers.anfis import get_anfis_score
import os
def prepare_segments_and_labels(signal, fs, r_peaks, annotation_sample, annotation_symbol, pre_R=0.2, post_R=0.4):
    """
    Vytvorí segmenty EKG signálu okolo R-vĺn a zodpovedajúce triedy.

    Parametre:
        signal (np.array): celý EKG signál
        fs (int): vzorkovacia frekvencia
        r_peaks (list): indexy detegovaných R-vĺn
        annotation_sample (list): všetky anotované pozície
        annotation_symbol (list): typy úderov pre dané anotácie
        pre_R, post_R (float): veľkosť segmentu okolo R-vlny (v sekundách)

    Výstup:
        X_segments (np.array): tvar (N, segment_length, 1)
        y_labels (list): triedy v stringoch ('Normal', 'Moderate', 'Severe')
    """
    from preprocessing.annotation_mapping import map_annotations_to_peaks

    # Segmentácia signálu
    segments = segment_heartbeats(signal, r_peaks, fs, pre_R, post_R)

    # Posun r_peaks kvôli segmentom (sú o 1 pozadu)
    adjusted_r_peaks = r_peaks[1:len(segments)+1]

    # Získanie tried podľa anotácií
    y_labels = map_annotations_to_peaks(adjusted_r_peaks, annotation_sample, annotation_symbol)

    # Filtrovanie len známych tried
    X_out = []
    y_out = []
    for seg, label in zip(segments, y_labels):
        if label != "Unknown":
            X_out.append(seg)
            y_out.append(label)

    return np.expand_dims(np.array(X_out), axis=-1), np.array(y_out)



################################

def run_classic_rf(df):
    print("\n🔎 Spúšťam klasický Random Forest (bez Fuzzy_Score)...")

    # Výber vstupných znakov a labelov
    X = df[["HR", "QRS_interval", "T_wave"]].values
    y = df["Label"].values

    # Vyváženie tried
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

    # Rozdelenie dát
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    # Trénovanie modelu
    model = train_random_forest(X_train, y_train, n_estimators=100)

    # Uloženie výsledkov
    os.makedirs("results/reports", exist_ok=True)
    evaluate_model(model, X_test, y_test, save_path="results/reports/rf_classic_predictions.csv")

    # Cross-validácia na pôvodných dátach
    cross_validate_model(
        train_random_forest, X, y,
        model_name="Random Forest",
        export_path="results/reports/rf_crossval_results.csv"
    )

def run_hybrid_rf(df):
    print("\n🧠 Spúšťam hybridný Random Forest (s Fuzzy_Score)...")
    #X = df[["HR", "QRS_interval", "T_wave", "Fuzzy_Score"]].values
    X = df[["HR", "QRS_interval", "T_wave", "Fuzzy_Score", "anfis_score"]].values
    y = df["Label"].values
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)
    model = train_random_forest(X_train, y_train, n_estimators=100)
    evaluate_model(model, X_test, y_test, save_path="results/reports/rf_hybrid_predictions.csv")
    cross_validate_model(
        train_random_forest, X, y,
        #model_name="Random Forest",
        model_name="RF + Fuzzy + ANFIS",
        export_path="results/reports/rf_fuzzy_anfis_crossval_results.csv"
        #export_path="results/reports/rffuzzy_crossval_results.csv"
    )

def run_classic_svm(df):
    print("\n🔎 Spúšťam klasický SVM (bez Fuzzy_Score)...")
    X = df[["HR", "QRS_interval", "T_wave"]].values
    y = df["Label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_res, y_res = SMOTE(random_state=42).fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

    model = train_svm(X_train, y_train, kernel='linear')  # Môžeš skúsiť aj 'rbf' alebo 'poly'
    evaluate_model(model, X_test, y_test, save_path="results/reports/svm_classic_scaled.csv")

def run_hybrid_svm(df):
    print("\n🧠 Spúšťam hybridný SVM (s Fuzzy_Score) – so škálovaním a linear kernel...")


    X = df[["HR", "QRS_interval", "T_wave", "Fuzzy_Score"]].values
    y = df["Label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_res, y_res = SMOTE(random_state=42).fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

    model = train_svm(X_train, y_train, kernel='linear')
    evaluate_model(model, X_test, y_test, save_path="results/reports/svm_hybrid_improved.csv")


def run_classic_dt(df):
    print("\n🔎 Spúšťam klasický Decision Tree (bez Fuzzy_Score)...")
    X = df[["HR", "QRS_interval", "T_wave"]].values
    y = df["Label"].values
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)
    model = train_decision_tree(X_train, y_train)
    evaluate_model(model, X_test, y_test, save_path="results/reports/dt_classic_predictions.csv")
    cross_validate_model(
        train_decision_tree, X, y,
        model_name="Decision Tree",
        export_path="results/reports/dt_crossval_results.csv"
    )


def run_hybrid_dt(df):
    print("\n🧠 Spúšťam hybridný Decision Tree (s Fuzzy_Score)...")
    X = df[["HR", "QRS_interval", "T_wave", "Fuzzy_Score"]].values
    y = df["Label"].values
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)
    model = train_decision_tree(X_train, y_train)
    evaluate_model(model, X_test, y_test, save_path="results/reports/dt_hybrid_predictions.csv")

    cross_validate_model(
        train_decision_tree, X, y,
        model_name="Decision Tree",
        export_path="results/reports/dtfuzzy_crossval_results.csv"
    )

def compare_models(df):
    # 🧠 Pridaj Fuzzy_Score len raz pre obe metódy
    print("compare models")
    df = df.copy()
    df["Fuzzy_Score"] = df.apply(lambda row: predict_arrhythmia(
        row["HR"], row["QRS_interval"], row["T_wave"], rule_based=False
    )[0], axis=1)
    df["anfis_score"] = df.apply(lambda row: get_anfis_score(
        row["HR"], row["QRS_interval"], row["T_wave"]
    ), axis=1)


    df = df.dropna()

    #run_classic_rf(df)
    run_hybrid_rf(df)
    #run_classic_dt(df)
   # run_hybrid_dt(df)

if __name__ == "__main__":
    all_features, all_labels, all_beat_counts_by_record = load_or_extract_fuzzy_features()

    summarize_loaded_beat_counts(all_beat_counts_by_record)
    df = pd.DataFrame(all_features)
    df["Label"] = all_labels

    #print("idem")
    compare_models(df)
    #run_cnn_models()




    """
    cross_validate_model(
        train_decision_tree, X, y,
        model_name="Decision Tree",
        export_path="results/reports/dt_crossval_results.csv"
    )
    """


