
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
from classifiers.cnn import build_1d_cnn, build_1d_cnn_with_fuzzy
from preprocessing.prepare_shared_cnn_dataset import prepare_shared_cnn_dataset
from config import REPORTS_DIR
def cross_validate_model(model_fn, X, y, model_name="Model", cv=5, export_path=None):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    all_reports = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🔁 Fold {fold}/{cv} - {model_name}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
        model = model_fn(X_res, y_res)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Flatten classification report
        report_flat = {
            f"{cls}_{metric}": val
            for cls, metrics in report.items()
            if isinstance(metrics, dict)
            for metric, val in metrics.items()
        }

        # Add top-level 'accuracy' metric
        if "accuracy" in report:
            report_flat["accuracy"] = report["accuracy"]

        report_flat["fold"] = fold
        all_reports.append(report_flat)

    results_df = pd.DataFrame(all_reports)
    avg_report = results_df.mean(numeric_only=True)

    # Show only available desired metrics
    desired_keys = ["accuracy", "macro avg_precision", "macro avg_recall", "macro avg_f1-score"]
    available_keys = [k for k in desired_keys if k in avg_report]

    print(f"\n📊 Priemerné metriky ({model_name}, {cv}-fold CV):")
    print(avg_report[available_keys])

    if export_path:
        results_df.to_csv(export_path, index=False)
        print(f"💾 Výsledky cross-validácie uložené do: {export_path}")

    return avg_report

def cross_validate_cnn_models(cv=5):
    print(f"\n🔁 Spúšťam {cv}-fold cross-validáciu pre klasickú a hybridnú CNN...")

    # Načítaj dáta
    X_segments, X_fuzzy, y, encoder = prepare_shared_cnn_dataset()

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    classic_scores = []
    hybrid_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_segments, y), 1):
        print(f"\n🔁 Fold {fold}/{cv}")

        X_seg_train, X_seg_test = X_segments[train_idx], X_segments[test_idx]
        X_fuzzy_train, X_fuzzy_test = X_fuzzy[train_idx], X_fuzzy[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        input_shape = X_seg_train.shape[1:]
        input_fuzzy_dim = X_fuzzy.shape[1]
        num_classes = len(np.unique(y))

        # Klasická CNN
        model_cnn = build_1d_cnn(input_shape=input_shape, num_classes=num_classes)
        model_cnn.fit(
            X_seg_train, y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=32,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=0
        )
        _, acc_cnn = model_cnn.evaluate(X_seg_test, y_test, verbose=0)
        classic_scores.append(acc_cnn)

        # Hybridná CNN
        model_hybrid = build_1d_cnn_with_fuzzy(
            input_shape_cnn=input_shape,
            input_shape_fuzzy=input_fuzzy_dim,
            num_classes=num_classes
        )
        model_hybrid.fit(
            [X_seg_train, X_fuzzy_train], y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=32,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=0
        )
        _, acc_hybrid = model_hybrid.evaluate([X_seg_test, X_fuzzy_test], y_test, verbose=0)
        hybrid_scores.append(acc_hybrid)

        print(f"✅ CNN presnosť: {acc_cnn:.4f} | Hybridná CNN: {acc_hybrid:.4f}")

    # Výsledky
    results = pd.DataFrame({
        "Fold": list(range(1, cv + 1)),
        "CNN_Accuracy": classic_scores,
        "Hybrid_CNN_Accuracy": hybrid_scores
    })
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "cnn_crossval_results.csv", index=False)
    print(f"\n💾 Výsledky CNN cross-validácie uložené do: {REPORTS_DIR / 'cnn_crossval_results.csv'}")

    print("\n📊 Priemerné presnosti:")
    print(results.mean(numeric_only=True))
