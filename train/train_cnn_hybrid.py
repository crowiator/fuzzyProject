import numpy as np
from sklearn.model_selection import train_test_split
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import load_model

from classifiers.cnn import build_1d_cnn_with_fuzzy
import numpy as np

from preprocessing.prepare_shared_cnn_dataset import prepare_shared_cnn_dataset
# 1. Príprava dát
X_segments, X_fuzzy, y_encoded, encoder = prepare_shared_cnn_dataset()

print("X_fuzzy min:", np.min(X_fuzzy, axis=0))
print("X_fuzzy max:", np.max(X_fuzzy, axis=0))
print("X_fuzzy any NaN:", np.isnan(X_fuzzy).any())
print("X_segments shape:", X_segments.shape)
# 2. Rozdelenie dát
X_seg_train, X_seg_test, X_fuzzy_train, X_fuzzy_test, y_train, y_test = train_test_split(
    X_segments, X_fuzzy, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# 3. Vytvorenie modelu
input_shape_cnn = X_seg_train.shape[1:]      # (segment_length, 1)
input_shape_fuzzy = X_fuzzy_train.shape[1]   # počet fuzzy čŕt
num_classes = len(np.unique(y_encoded))

model = build_1d_cnn_with_fuzzy(input_shape_cnn, input_shape_fuzzy, num_classes)

# 4. Tréning
callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
model.fit(
    [X_seg_train, X_fuzzy_train], y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=callbacks
)

# 5. Vyhodnotenie
loss, acc = model.evaluate([X_seg_test, X_fuzzy_test], y_test)
print(f"\n✅ Test accuracy: {acc:.4f}")

# 6. Uloženie modelu
model.save("results/models/cnn_hybrid_model.h5")
print("💾 Hybridný CNN model uložený do results/models/cnn_hybrid_model.h5")
