"""
Voice Emotion Detector - Model Trainer
Trains a Keras CNN model on RAVDESS dataset using 40 MFCC features.
Run: python model_trainer.py <path_to_ravdess_dataset>
"""

import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def extract_features(file_path, sr=22050, duration=3.0, offset=0.0):
    """
    Extract 40 MFCC features — must match create_features.py (kaiser_fast, offset=0.0).
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration, offset=offset, res_type='kaiser_fast')
        if len(audio) < 1000:
            return None
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs.astype(np.float32)
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


def augment_audio(file_path, sr=22050, duration=3.0):
    """Generate augmented versions of an audio file."""
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration, offset=0.5)
        if len(audio) < 1000:
            return []

        augments = []

        # Time stretch (±10%)
        for rate in [0.9, 1.1]:
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            if len(stretched) >= sr:
                stretched = stretched[:sr]
            else:
                stretched = np.pad(stretched, (0, sr - len(stretched)))
            augments.append(stretched)

        # Pitch shift (±2 semitones)
        for steps in [-2, 2]:
            pitched = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=steps)
            augments.append(pitched)

        # Add noise
        noise_factor = 0.005
        noisy = audio + noise_factor * np.random.randn(len(audio))
        augments.append(noisy)

        return augments
    except Exception as e:
        return []


EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

LABEL_CONVERSION = {
    0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
    4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
}

def load_ravdess_data(dataset_path, augment=True):
    """Load all RAVDESS audio files and extract features with augmentation."""
    X, y = [], []
    total = 0
    errors = 0

    print(f"\n[OK] Scanning dataset at: {dataset_path}")

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in EMOTION_MAP:
                        file_path = os.path.join(root, file)

                        # Original
                        features = extract_features(file_path)
                        if features is not None:
                            X.append(features)
                            y.append(EMOTION_MAP[emotion_code])
                            total += 1

                        # Augmented versions (4x data)
                        if augment:
                            augments = augment_audio(file_path)
                            for aug_audio in augments:
                                try:
                                    aug_mfccs = np.mean(
                                        librosa.feature.mfcc(y=aug_audio, sr=22050, n_mfcc=40).T,
                                        axis=0
                                    ).astype(np.float32)
                                    X.append(aug_mfccs)
                                    y.append(EMOTION_MAP[emotion_code])
                                    total += 1
                                except:
                                    pass

                        if total % 200 == 0:
                            print(f"  Processed {total} samples (with augmentation)...", end='\r')

    print(f"\n[OK] Loaded {total} samples (with augmentation). Errors: {errors}")
    return np.array(X), np.array(y)


def build_model(input_shape=(40, 1), num_classes=8):
    """Build an improved CNN model with residual-style blocks."""
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Conv1D, Activation, Dropout, MaxPooling1D,
                                         Flatten, Dense, BatchNormalization)
    from tensorflow.keras.regularizers import l2

    model = Sequential([
        # Block 1
        Conv1D(256, 3, padding='same', input_shape=input_shape, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        MaxPooling1D(pool_size=2),

        # Block 2
        Conv1D(128, 3, padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        MaxPooling1D(pool_size=2),

        # Block 3
        Conv1D(64, 3, padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),

        Flatten(),
        Dense(128, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(dataset_path, model_save_path='models/'):
    """Train a Keras CNN model for emotion detection."""
    os.makedirs(model_save_path, exist_ok=True)

    X, y = load_ravdess_data(dataset_path, augment=True)
    if len(X) == 0:
        print("[FAIL] No data found! Check dataset path.")
        return

    print(f"\n[DATA] Dataset: {len(X)} samples, {X.shape[1]} features")

    from collections import Counter
    counts = Counter(y)
    for emotion, count in sorted(counts.items()):
        print(f"  {emotion}: {count}")

    label_map = {v: k for k, v in LABEL_CONVERSION.items()}
    y_encoded = np.array([label_map[label] for label in y])

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Class weights for imbalanced data
    class_counts = Counter(y_train)
    total = len(y_train)
    class_weight = {
        i: total / (len(class_counts) * class_counts[i])
        for i in range(8)
    }
    print(f"\n[WEIGHT]  Class weights: {class_weight}")

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    num_classes = 8
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    X_train_reshaped = X_train_scaled.reshape(-1, 40, 1).astype(np.float32)
    X_test_reshaped = X_test_scaled.reshape(-1, 40, 1).astype(np.float32)

    model = build_model(input_shape=(40, 1), num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    print("\n[TRAIN] Training CNN model...")
    history = model.fit(
        X_train_reshaped, y_train_cat,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
    print(f"\n[ACC] Test Accuracy: {acc * 100:.2f}%")

    # Per-class accuracy
    y_pred = model.predict(X_test_reshaped, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test_cat, axis=1)

    from sklearn.metrics import classification_report
    target_names = [LABEL_CONVERSION[i] for i in range(8)]
    print("\n[REPORT] Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=target_names))

    model.save(os.path.join(model_save_path, 'Emotion_Voice_Detection_Model.h5'))

    # Save scaler for inference
    import joblib
    joblib.dump(scaler, os.path.join(model_save_path, 'scaler.pkl'))

    print(f"\n[SAVE] Model saved to '{model_save_path}/Emotion_Voice_Detection_Model.h5'")
    print("[OK] Training complete! Run app.py to use the model.")

    return acc


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python model_trainer.py <path_to_ravdess_dataset>")
        print("\nExample:")
        print("  python model_trainer.py ./RAVDESS")
    else:
        dataset_path = sys.argv[1]
        if not os.path.exists(dataset_path):
            print(f"[FAIL] Path not found: {dataset_path}")
        else:
            train_model(dataset_path, model_save_path='models/')