import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from classifiers.cnn import build_1d_cnn, build_1d_cnn_with_fuzzy
from preprocessing.prepare_shared_cnn_dataset import prepare_shared_cnn_dataset
from config import REPORTS_DIR
from crossval import cross_validate_cnn_models
def run_cnn_models():
    print("\n🧠 Spúšťam klasickú a hybridnú CNN...")

    # 1. Načítanie segmentov a fuzzy vstupov
    X_segments, X_fuzzy, y_encoded, encoder = prepare_shared_cnn_dataset()

    # 2. Rozdelenie rovnakým spôsobom pre oba modely
    X_seg_train, X_seg_test, X_fuzzy_train, X_fuzzy_test, y_train, y_test = train_test_split(
        X_segments, X_fuzzy, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    input_shape = X_seg_train.shape[1:]
    input_fuzzy_dim = X_fuzzy.shape[1]
    num_classes = len(np.unique(y_encoded))

    # 3. Tréning klasickej CNN
    print("\n🔎 Trénujem klasickú CNN")
    model_cnn = build_1d_cnn(input_shape=input_shape, num_classes=num_classes)
    model_cnn.fit(
        X_seg_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=0
    )
    loss_cnn, acc_cnn = model_cnn.evaluate(X_seg_test, y_test, verbose=0)

    # 4. Tréning hybridnej CNN
    print("\n🧪 Trénujem hybridnú CNN (s fuzzy vstupmi)")
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
    loss_hybrid, acc_hybrid = model_hybrid.evaluate([X_seg_test, X_fuzzy_test], y_test, verbose=0)

    # 5. Výpis a uloženie
    print(f"\n✅ Accuracy klasickej CNN: {acc_cnn:.4f}")
    print(f"✅ Accuracy hybridnej CNN: {acc_hybrid:.4f}")

    results = pd.DataFrame([
        ["CNN", "Classic", acc_cnn],
        ["CNN", "Hybrid", acc_hybrid]
    ], columns=["Model", "Typ", "Presnosť"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "cnn_results.csv", index=False)
    print(f"💾 Výsledky CNN uložené do: {REPORTS_DIR / 'cnn_results.csv'}")
    cross_validate_cnn_models(cv=5)