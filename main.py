import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib
from sklearn.metrics import precision_score, recall_score, f1_score
matplotlib.use('TkAgg')
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from classifiers.fuzzy_classifier import FuzzyClassifier
from classifiers.traditional_models import (
    train_knn,
    train_svm,
    train_decision_tree,
    train_random_forest,
    evaluate_model,
)
from sklearn.model_selection import cross_val_score

import pandas as pd
from preprocessing.load import summarize_loaded_beat_counts
from preprocessing.fuzzy_feature_loader import load_or_extract_fuzzy_features, load_or_extract_fuzzy_features2


def run_fuzzy_classificator():
    features, labels, beat_counts = load_or_extract_fuzzy_features()
    fuzzy_classifier = FuzzyClassifier()

    results = []

    for feature, label in zip(features, labels):
        hr = feature["HR"]
        qrs = feature["QRS_interval"]
        twa = feature["T_wave"]

        try:
            fuzzy_score, fuzzy_label, memberships = fuzzy_classifier.predict(hr, qrs, twa)
        except Exception as e:
            print(f"Chyba klasifikácie úderu: HR={hr}, QRS={qrs}, TWA={twa} → {e}")
            fuzzy_score, fuzzy_label, memberships = (None, "Error", {"normal": 0, "moderate": 0, "severe": 0})

        results.append({
            "HR": hr,
            "QRS_interval": qrs,
            "T_wave": twa,
            "True_Label": label,
            "Fuzzy_Score": fuzzy_score,
            "Fuzzy_Label": fuzzy_label,
            "Membership_Normal": memberships["normal"],
            "Membership_Moderate": memberships["moderate"],
            "Membership_Severe": memberships["severe"]
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv("results/fuzzy_classification_results.csv", index=False)

    print("Fuzzy classification complete. Results saved to 'results/fuzzy_classification_results.csv'.")

    summarize_loaded_beat_counts(beat_counts)


def fuzzy_classificator_statistic():
    file_path = 'results/fuzzy_classification_results.csv'

    df = pd.read_csv(file_path)
    df_valid = df[df["Fuzzy_Label"] != "Invalid"]

    print("Total number of beats:", len(df))
    print("Počet validných úderov použitých v analýze:", len(df_valid))

    print("\nDistribution by fuzzy classifier:")
    print(df_valid["Fuzzy_Label"].value_counts())
    print("\nDistribution by true annotations:")
    print(df_valid["True_Label"].value_counts())

    print("\nAverage feature values (valid only):")
    print(f"- Average HR: {df_valid['HR'].mean():.2f} bpm")
    print(f"- Average QRS interval: {df_valid['QRS_interval'].mean():.2f} ms")
    print(f"- Average T-wave amplitude: {df_valid['T_wave'].mean():.3f} mV")

    cm = confusion_matrix(
        df_valid["True_Label"],
        df_valid["Fuzzy_Label"],
        labels=["Normal", "Moderate", "Severe"]
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Moderate", "Severe"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix pre Fuzzy Klasifikátor")
    plt.show()

    report = classification_report(
        df_valid["True_Label"],
        df_valid["Fuzzy_Label"],
        labels=["Normal", "Moderate", "Severe"],
        target_names=["Normal", "Moderate", "Severe"]
    )

    print(report)


def find_invalid_features():
    file_path = 'results/fuzzy_classification_results.csv'
    df = pd.read_csv(file_path)

    invalid_cases = df[df["Fuzzy_Label"] == "Invalid"]
    print(f"Total invalid cases: {len(invalid_cases)}\n")

    invalid_hr = invalid_cases[(invalid_cases["HR"] < 40) | (invalid_cases["HR"] > 120)]
    print(f"Invalid HR cases: {len(invalid_hr)} ({len(invalid_hr) / len(invalid_cases) * 100:.2f}%)")

    invalid_qrs = invalid_cases[(invalid_cases["QRS_interval"] < 50) | (invalid_cases["QRS_interval"] > 120)]
    print(f"Invalid QRS_interval cases: {len(invalid_qrs)} ({len(invalid_qrs) / len(invalid_cases) * 100:.2f}%)")

    invalid_twa = invalid_cases[(invalid_cases["T_wave"] <= 0.0) | (invalid_cases["T_wave"] > 0.6)]
    print(f"Invalid T_wave cases: {len(invalid_twa)} ({len(invalid_twa) / len(invalid_cases) * 100:.2f}%)")


def preparing_hybrid_features():
    print("loading")
    features, labels, beat_counts = load_or_extract_fuzzy_features()
    df = pd.DataFrame(features)
    df["Label"] = labels
    fuzzy_classifier = FuzzyClassifier()

    def extract_hybrid_features(row):
        fuzzy_score, fuzzy_label, memberships = fuzzy_classifier.predict(
            row["HR"], row["QRS_interval"], row["T_wave"]
        )
        return pd.Series({
            "Fuzzy_Score": fuzzy_score,
            "Fuzzy_μ_Normal": memberships["normal"],
            "Fuzzy_μ_Moderate": memberships["moderate"],
            "Fuzzy_μ_Severe": memberships["severe"],
            "Fuzzy_Label": fuzzy_label
        })

    hybrid_features = df.apply(extract_hybrid_features, axis=1)
    df_hybrid = pd.concat([df, hybrid_features], axis=1)
    df_hybrid = df_hybrid.dropna(subset=["Fuzzy_Score"])
    df_hybrid = df_hybrid[df_hybrid["Fuzzy_Label"] != "Invalid"]
    print("done")
    # Spoločné filtrovanie pre oba typy modelov
    df_filtered = df_hybrid.dropna(subset=["Fuzzy_Score"])
    df_filtered = df_filtered[df_filtered["Fuzzy_Label"] != "Invalid"]
    print(df_hybrid)
    print(df_filtered)
    return df_hybrid


def run_classical_algorithm(df):
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    df = df.dropna(subset=["HR", "QRS_interval", "T_wave", "Label"])
    X = df[["HR", "QRS_interval", "T_wave"]].values
    y = df["Label"].values

    X_resampled, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE
    )

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    print("\n Decision Tree:")
    dt_model = train_decision_tree(X_train, y_train, criterion='entropy', max_depth=None, max_features=None,
                                   min_samples_leaf=1, min_samples_split=2)
    evaluate_model(dt_model, X_test, y_test, save_path="results/reports/traditional/traditional", model_name="DecisionTree")
    dt_scores = cross_val_score(dt_model, X_resampled, y_resampled, cv=CV_FOLDS)
    print(f" Decision Tree cross-validation accuracy: {np.mean(dt_scores):.4f}")

    y_pred_dt = dt_model.predict(X_test)
    dt_precision = precision_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    dt_recall = recall_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    dt_f1 = f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    accuracies.append(np.mean(dt_scores))
    precisions.append(dt_precision)
    recalls.append(dt_recall)
    f1s.append(dt_f1)



    print("\n Random Forest:")
    rf_model = train_random_forest(X_train, y_train, criterion='entropy', max_depth=20, max_features=None,
                                   min_samples_leaf=1, min_samples_split=5, n_estimators=150, )
    evaluate_model(rf_model, X_test, y_test, save_path="results/reports/traditional/traditional", model_name="RandomForest")
    rf_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=CV_FOLDS)
    print(f" Random Forest cross-validation accuracy: {np.mean(rf_scores):.4f}")

    y_pred_rf = rf_model.predict(X_test)
    rf_precision = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    rf_recall = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    rf_f1 = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    accuracies.append(np.mean(rf_scores))
    precisions.append(rf_precision)
    recalls.append(rf_recall)
    f1s.append(rf_f1)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
        X_scaled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE
    )
    print("\n Support Vector Machine (SVM):")
    svm_model = train_svm(X_train_svm, y_train_svm, C=10, degree=2, gamma='auto', kernel='rbf')
    evaluate_model(svm_model, X_test_svm, y_test_svm, save_path="results/reports/traditional/traditional",
                   model_name="SVM")
    svm_scores = cross_val_score(svm_model, X_scaled, y_resampled, cv=CV_FOLDS)
    print(f" SVM cross-validation accuracy: {np.mean(svm_scores):.4f}")
    y_pred_svm = svm_model.predict(X_test_svm)
    svm_precision = precision_score(y_test_svm, y_pred_svm, average='weighted', zero_division=0)
    svm_recall = recall_score(y_test_svm, y_pred_svm, average='weighted', zero_division=0)
    svm_f1 = f1_score(y_test_svm, y_pred_svm, average='weighted', zero_division=0)
    accuracies.append(np.mean(svm_scores))
    precisions.append(svm_precision)
    recalls.append(svm_recall)
    f1s.append(svm_f1)

    print("\n k-Nearest Neighbors (kNN):")
    knn_model = train_knn(X_train_svm, y_train_svm, metric='manhattan', n_neighbors=5, weights='distance')
    evaluate_model(knn_model, X_test_svm, y_test_svm, save_path="results/reports/traditional/traditional",
                   model_name="kNN")
    knn_scores = cross_val_score(knn_model, X_scaled, y_resampled, cv=CV_FOLDS)
    print(f" kNN cross-validation accuracy: {np.mean(knn_scores):.4f}")
    y_pred_knn = knn_model.predict(X_test_svm)
    knn_precision = precision_score(y_test_svm, y_pred_knn, average='weighted', zero_division=0)
    knn_recall = recall_score(y_test_svm, y_pred_knn, average='weighted', zero_division=0)
    knn_f1 = f1_score(y_test_svm, y_pred_knn, average='weighted', zero_division=0)
    accuracies.append(np.mean(knn_scores))
    precisions.append(knn_precision)
    recalls.append(knn_recall)
    f1s.append(knn_f1)
    return accuracies, precisions, recalls, f1s




def run_hybrid_models(df_hybrid):
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5

    X = df_hybrid[[
        "HR", "QRS_interval", "T_wave",
        "Fuzzy_Score", "Fuzzy_μ_Normal", "Fuzzy_μ_Moderate", "Fuzzy_μ_Severe"
    ]].values
    y = df_hybrid["Label"].values

    X_resampled, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE
    )

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    print("\n Hybrid Decision Tree:")
    dt_model = train_decision_tree(X_train, y_train, criterion='entropy', max_depth=None, max_features=None,
                                   min_samples_leaf=1, min_samples_split=2)
    evaluate_model(dt_model, X_test, y_test, save_path="results/reports/hybrid/hybrid", model_name="DecisionTree")
    dt_scores = cross_val_score(dt_model, X_resampled, y_resampled, cv=CV_FOLDS)
    print(f" Hybrid Decision Tree cross-validation accuracy: {np.mean(dt_scores):.4f}")
    y_pred_dt = dt_model.predict(X_test)
    dt_precision = precision_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    dt_recall = recall_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    dt_f1 = f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    accuracies.append(np.mean(dt_scores))
    precisions.append(dt_precision)
    recalls.append(dt_recall)
    f1s.append(dt_f1)


    print("\n Hybrid Random Forest:")
    rf_model = train_random_forest(X_train, y_train, criterion='entropy', max_depth=None, max_features=None,
                                   min_samples_leaf=1, min_samples_split=2, n_estimators=200)
    evaluate_model(rf_model, X_test, y_test, save_path="results/reports/hybrid/hybrid", model_name="RandomForest")
    rf_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=CV_FOLDS)
    print(f" Hybrid Random Forest cross-validation accuracy: {np.mean(rf_scores):.4f}")
    y_pred_rf = rf_model.predict(X_test)
    rf_precision = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    rf_recall = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    rf_f1 = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    accuracies.append(np.mean(rf_scores))
    precisions.append(rf_precision)
    recalls.append(rf_recall)
    f1s.append(rf_f1)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
        X_scaled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE
    )
    print("\n Hybrid Support Vector Machine (SVM):")
    svm_model = train_svm(X_train_svm, y_train_svm, C=10, degree=2, gamma='auto', kernel='rbf')
    evaluate_model(svm_model, X_test_svm, y_test_svm, save_path="results/reports/hybrid/hybrid",
                   model_name="SVM")
    svm_scores = cross_val_score(svm_model, X_scaled, y_resampled, cv=CV_FOLDS)
    print(f" SVM cross-validation accuracy: {np.mean(svm_scores):.4f}")
    y_pred_svm = svm_model.predict(X_test_svm)
    svm_precision = precision_score(y_test_svm, y_pred_svm, average='weighted', zero_division=0)
    svm_recall = recall_score(y_test_svm, y_pred_svm, average='weighted', zero_division=0)
    svm_f1 = f1_score(y_test_svm, y_pred_svm, average='weighted', zero_division=0)
    accuracies.append(np.mean(svm_scores))
    precisions.append(svm_precision)
    recalls.append(svm_recall)
    f1s.append(svm_f1)


    print("\n Hybrid k-Nearest Neighbors (kNN):")
    knn_model = train_knn(X_train_svm, y_train_svm, metric='manhattan', n_neighbors=3, weights='distance')
    evaluate_model(knn_model, X_test_svm, y_test_svm, save_path="results/reports/hybrid/hybrid", model_name="kNN")
    knn_scores = cross_val_score(knn_model, X_scaled, y_resampled, cv=CV_FOLDS)
    print(f" Hybrid kNN cross-validation accuracy: {np.mean(knn_scores):.4f}")
    y_pred_knn = knn_model.predict(X_test_svm)
    knn_precision = precision_score(y_test_svm, y_pred_knn, average='weighted', zero_division=0)
    knn_recall = recall_score(y_test_svm, y_pred_knn, average='weighted', zero_division=0)
    knn_f1 = f1_score(y_test_svm, y_pred_knn, average='weighted', zero_division=0)
    accuracies.append(np.mean(knn_scores))
    precisions.append(knn_precision)
    recalls.append(knn_recall)
    f1s.append(knn_f1)

    return accuracies, precisions, recalls, f1s




def compare_models(
    classical_results, hybrid_results,
    classical_precision, hybrid_precision,
    classical_recall, hybrid_recall,
    classical_f1, hybrid_f1
):
    comparison = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest", "SVM", "kNN"],
        "Classical Accuracy": classical_results,
        "Hybrid Accuracy": hybrid_results,
        "Classical Precision": classical_precision,
        "Hybrid Precision": hybrid_precision,
        "Classical Recall": classical_recall,
        "Hybrid Recall": hybrid_recall,
        "Classical F1-Score": classical_f1,
        "Hybrid F1-Score": hybrid_f1
    })

    print("\nComparison of models:")
    print(comparison)

    # Ensure directory exists
    os.makedirs('results/reports', exist_ok=True)

    # Save Excel - správne bez open
    comparison.to_excel('results/reports/comparison_table.xlsx', index=False)

    # Optional: Save CSV - ak chceš aj csv verziu
    with open('results/reports/comparison_table.csv', mode='w', newline='') as file:
        comparison.to_csv(file, index=False)

    # Draw graphs
    comparison.plot(x="Model", y=["Classical Accuracy", "Hybrid Accuracy"], kind="bar")
    plt.title("Porovnanie: Klasické vs Hybridné modely - Presnosť (Accuracy)")
    plt.ylabel("Presnosť (Accuracy)")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/reports/model_comparison_accuracy.png")
    plt.show()

    comparison.plot(x="Model", y=["Classical F1-Score", "Hybrid F1-Score"], kind="bar")
    plt.title("Porovnanie: Klasické vs Hybridné modely - F1 skóre (F1-Score)")
    plt.ylabel("F1 skóre (F1-Score)")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/reports/model_comparison_f1.png")
    plt.show()

    comparison.plot(x="Model", y=["Classical Precision", "Hybrid Precision"], kind="bar")
    plt.title("Porovnanie: Klasické vs Hybridné modely - Presnosť (Precision)")
    plt.ylabel("Presnosť (Precision)")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/reports/model_comparison_precision.png")
    plt.show()

    comparison.plot(x="Model", y=["Classical Recall", "Hybrid Recall"], kind="bar")
    plt.title("Porovnanie: Klasické vs Hybridné modely - Úplnosť (Recall)")
    plt.ylabel("Úplnosť (Recall)")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/reports/model_comparison_recall.png")
    plt.show()


# Inicializácia
if __name__ == "__main__":
    """
    print("\n--- Loading data for hybrid and traditional models ---")
    df_hybrid = preparing_hybrid_features()

    # Count original and filtered data
    total_original = len(load_or_extract_fuzzy_features()[0])
    filtered_count = len(df_hybrid)

    print("\nClass distribution after fuzzy filtering:")
    print(df_hybrid["Label"].value_counts())
    print(f"\nTotal original beats: {total_original}")
    print(f"Beats used after fuzzy filtering: {filtered_count}")
    print(f"Filtered out beats: {total_original - filtered_count} "
          f"({(total_original - filtered_count) / total_original * 100:.2f}%)")

    print("\n--- Running hybrid models ---")
    hybrid_acc, hybrid_prec, hybrid_rec, hybrid_f1 = run_hybrid_models(df_hybrid)

    print("\n--- Running traditional models ---")
    classical_acc, classical_prec, classical_rec, classical_f1 = run_classical_algorithm(df_hybrid)

    compare_models(classical_acc, hybrid_acc,
                   classical_prec, hybrid_prec,
                   classical_rec, hybrid_rec,
                   classical_f1, hybrid_f1)
    # run_fuzzy_classificator()
    # fuzzy_classificator_statistic()
    # find_invalid_features()
    """
    load_or_extract_fuzzy_features2()
