import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

from classifiers.cnn import build_1d_cnn
from preprocessing.segmentation import segment_heartbeats
from preprocessing.annotation_mapping import map_annotations_to_peaks
from preprocessing.load import load_mitbih_record
from config import RECORD_NAMES
from preprocessing.prepare_shared_cnn_dataset import prepare_shared_cnn_dataset
# Parametre segmentácie
PRE_R = 0.2
POST_R = 0.4

X_segments, X_fuzzy, y_encoded, encoder = prepare_shared_cnn_dataset()


# Rozdelenie dát
X_train, X_test, y_train, y_test = train_test_split(
    X_segments, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Tréning modelu
model = build_1d_cnn(input_shape=X_train.shape[1:], num_classes=len(np.unique(y_encoded)))
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

# Uloženie modelu
model.save("../results/models/cnn_ecg_model.h5")
print("💾 Klasický CNN model uložený do results/models/cnn_ecg_model.h5")
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test accuracy: {acc:.4f}")
np.save("../results/models/cnn_label_encoder_classes.npy", encoder.classes_)
