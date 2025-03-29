import matplotlib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
matplotlib.use("TkAgg")
from imblearn.over_sampling import SMOTE
from preprocessing.load import load_mitbih_record, summarize_loaded_beat_counts
from preprocessing.filtering import (
    lowpass_filter,
    dwt_filtering,
    normalize_zscore
)
from preprocessing.feature_extraction import extract_features_for_fuzzy
from preprocessing.qrs_processing import (
    differentiate,
    squaring,
    moving_window_integration,
    detect_r_peaks
)
from preprocessing.annotation_mapping import map_annotations_to_peaks
from config import RECORD_NAMES
from classifiers.traditional_models import train_decision_tree, train_random_forest, train_svm, evaluate_model
from classifiers.fuzzy_classifier import predict_arrhythmia
from utils.visualization import plot_signal_comparison

def run_classic_rf(df):
    print("\n🔎 Spúšťam klasický Random Forest (bez Fuzzy_Score)...")
    X = df[["HR", "QRS_interval", "T_wave"]].values
    y = df["Label"].values
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)
    model = train_random_forest(X_train, y_train, n_estimators=100)
    evaluate_model(model, X_test, y_test, save_path="results/reports/rf_classic_predictions.csv")

def run_hybrid_rf(df):
    print("\n🧠 Spúšťam hybridný Random Forest (s Fuzzy_Score)...")
    X = df[["HR", "QRS_interval", "T_wave", "Fuzzy_Score"]].values
    y = df["Label"].values
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)
    model = train_random_forest(X_train, y_train, n_estimators=100)
    evaluate_model(model, X_test, y_test, save_path="results/reports/rf_hybrid_predictions.csv")

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


def run_hybrid_dt(df):
    print("\n🧠 Spúšťam hybridný Decision Tree (s Fuzzy_Score)...")
    X = df[["HR", "QRS_interval", "T_wave", "Fuzzy_Score"]].values
    y = df["Label"].values
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)
    model = train_decision_tree(X_train, y_train)
    evaluate_model(model, X_test, y_test, save_path="results/reports/dt_hybrid_predictions.csv")

def compare_models(df):
    # 🧠 Pridaj Fuzzy_Score len raz pre obe metódy
    print("compare models")
    df = df.copy()
    df["Fuzzy_Score"] = df.apply(lambda row: predict_arrhythmia(
        row["HR"], row["QRS_interval"], row["T_wave"], rule_based=False
    )[0], axis=1)

    df = df.dropna()

    run_classic_rf(df)
    run_hybrid_rf(df)
    run_classic_svm(df)
    run_hybrid_svm(df)
    run_classic_dt(df)
    run_hybrid_dt(df)

if __name__ == "__main__":
    record_names = RECORD_NAMES
    all_features = []
    all_labels = []
    all_beat_counts_by_record = {}

    for record in record_names:
        try:
            signal, fs, r_peak_positions, beat_types, beat_counts, annotation = load_mitbih_record(
                record_name=record, path="./data/mit/")
            all_beat_counts_by_record[record] = beat_counts

            filtered_signal = lowpass_filter(signal, fs)
            filtered_signal = dwt_filtering(filtered_signal)
            signal_for_amplitude = filtered_signal

            detected_r_peaks = r_peak_positions
            features = extract_features_for_fuzzy(signal_for_amplitude, fs, detected_r_peaks)
            detected_r_peaks = detected_r_peaks[1:]
            labels = map_annotations_to_peaks(detected_r_peaks, annotation.sample, annotation.symbol)

            for feat, label in zip(features, labels):
                if label != "Unknown":
                    all_features.append(feat)
                    all_labels.append(label)

            print(f"✅ Spracovaný záznam {record}: {len(features)} úderov")

        except Exception as e:
            print(f"⚠️ Chyba pri zázname {record}: {e}")

    summarize_loaded_beat_counts(all_beat_counts_by_record)
    df = pd.DataFrame(all_features)
    df["Label"] = all_labels

    # 🔀 Porovnanie klasického a hybridného modelu
    print("idem")
    compare_models(df)
