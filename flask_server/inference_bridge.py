"""
StutterSense — Flask Inference Bridge
======================================
Loads the trained Transformer model + scaler and exposes a prediction
method used by app.py.

FIX LOG
-------
- delta/delta2 computation now correctly transposes before librosa.feature.delta
  to match the training pipeline in test_model.py / build_wav_dataset.py.
- Scaler is applied frame-wise: (T, F) → (T*1, F) → (T, F), not on the full
  flattened sequence at once.
- SEQ_LEN and HOP_LENGTH are explicit constants synced with training.
"""

import os
import pickle
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras

# Suppress noisy TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Constants (must match test_model.py / build_wav_dataset.py) ───────────────
SR         = 16_000
SEQ_LEN    = 128
HOP_LENGTH = 512
N_MFCC     = 13
N_FEATURES = 39   # 13 + 13Δ + 13ΔΔ
CLASS_NAMES = [
    "NoStutter", "Prolongation", "Block",
    "SoundRepetition", "WordRepetition"
]


# ── Custom loss required for model de-serialisation ───────────────────────────
class FocalLoss(keras.losses.Loss):
    """Focal Loss — must match the definition used during training."""
    def __init__(self, gamma: float = 1.5, alpha: float = 0.25,
                 name: str = "focal_loss", **kwargs):
        # **kwargs absorbs Keras internal keys like 'reduction'
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true  = tf.cast(y_true, tf.int32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce      = -tf.math.log(y_pred)
        one_hot = tf.one_hot(y_true, depth=len(CLASS_NAMES))
        p_t     = tf.reduce_sum(y_pred * one_hot, axis=-1)
        focal_w = self.alpha * tf.pow(1.0 - p_t, self.gamma)
        loss    = focal_w * tf.reduce_sum(ce * one_hot, axis=-1)
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg


# ── Feature extraction (matches training pipeline exactly) ────────────────────
def _extract_features(audio_array: np.ndarray) -> np.ndarray:
    """
    Convert a raw audio window (1-D numpy array at 16 kHz) into a
    (SEQ_LEN, N_FEATURES) = (128, 39) feature tensor.

    Pipeline mirrors test_model.py and build_wav_dataset.py:
      mfcc (T,13) → delta (T,13) → delta2 (T,13) → concat (T,39) → pad/truncate
    """
    # 1. Base 13 MFCCs  → shape (T, 13)
    mfcc = librosa.feature.mfcc(
        y=audio_array, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    ).T

    # 2. Delta / Delta-Delta
    #    librosa.feature.delta expects (features, frames) so we transpose first
    delta  = librosa.feature.delta(mfcc.T, order=1).T   # (T, 13)
    delta2 = librosa.feature.delta(mfcc.T, order=2).T   # (T, 13)

    # 3. Concatenate → (T, 39)
    feats = np.concatenate([mfcc, delta, delta2], axis=-1)

    # 4. Pad or truncate to exactly SEQ_LEN frames
    T = feats.shape[0]
    if T >= SEQ_LEN:
        feats = feats[:SEQ_LEN]
    else:
        feats = np.pad(feats, ((0, SEQ_LEN - T), (0, 0)), mode='constant')

    return feats.astype(np.float32)   # (128, 39)


# ── Main inference class ───────────────────────────────────────────────────────
class StutterInferenceBridge:
    def __init__(self,
                 model_path: str = "transformer_model.keras",
                 scaler_path: str = "scaler.pkl"):
        """Loads the model and scaler once at server start."""
        print("[INFO] Loading StutterSense AI Engine...")

        self.model = keras.models.load_model(
            model_path,
            custom_objects={"FocalLoss": FocalLoss}
        )
        with open(scaler_path, "rb") as fh:
            self.scaler = pickle.load(fh)

        print("[INFO] AI Engine ready!")

    def predict_audio_chunk(self, audio_array: np.ndarray) -> dict:
        """
        Classify a raw audio chunk (numpy array of float32 @ 16 kHz).

        Returns a dict:
          { "prediction": str, "confidence": float 0-100, "status": "success" }
          or { "status": "error", "message": str }
        """
        try:
            feats = _extract_features(audio_array)           # (128, 39)

            # Scale frame-wise using the training scaler
            feats = self.scaler.transform(
                feats.reshape(-1, N_FEATURES)
            ).reshape(1, SEQ_LEN, N_FEATURES)                # (1, 128, 39)

            probs    = self.model.predict(feats, verbose=0)[0]  # (5,)
            pred_idx = int(np.argmax(probs))
            conf     = float(probs[pred_idx])

            # ── Sensitivity Boost: surface stutter predictions ──────
            if pred_idx == 0:
                alt_idx = int(np.argmax(probs[1:])) + 1
                if probs[alt_idx] > 0.15:
                    pred_idx = alt_idx
                    conf = float(probs[pred_idx])

            return {
                "prediction": CLASS_NAMES[pred_idx],
                "confidence": round(conf * 100, 2),
                "all_probs":  {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                               for i in range(len(CLASS_NAMES))},
                "status": "success"
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}