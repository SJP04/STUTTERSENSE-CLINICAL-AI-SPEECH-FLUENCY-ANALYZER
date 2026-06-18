"""
StutterSense — Dataset Builder (SEP-28k)
==========================================
Reads the SEP-28k CSV labels, finds the corresponding .wav files, extracts
(SEQ_LEN=128, 39) feature tensors and saves them as a compressed .npz file
ready for training in test_model.py.

FIX LOG
-------
- delta / delta2 now computed with mfcc.T → librosa.feature.delta → .T
  (was previously applied directly on (T,13) matrix, producing wrong shapes)
- Added explicit dtype float32 for mfcc before delta to prevent type errors.
- Moved constants to top-level so they can be imported by other scripts.
"""

import os
import numpy as np
import pandas as pd
import librosa

# ── Constants (must match test_model.py and inference_bridge.py) ───────────────
SR         = 16_000
SEQ_LEN    = 128
HOP_LENGTH = 512
N_MFCC     = 13
N_FEATURES = 39
TARGET_COLS = ['NoStutteredWords', 'Prolongation', 'Block', 'SoundRep', 'WordRep']


def process_wav_to_tensor(file_path: str) -> np.ndarray | None:
    """
    Convert a single .wav file into a (SEQ_LEN, N_FEATURES) = (128, 39) tensor.

    Pipeline:
      load @16kHz → MFCC(13) → Δ(13) → ΔΔ(13) → concat(39) → pad/truncate(128)
    """
    try:
        y, _ = librosa.load(file_path, sr=SR, duration=3.0, mono=True)

        # Base MFCCs: librosa returns (n_mfcc, frames) → transpose to (T, 13)
        mfcc = librosa.feature.mfcc(
            y=y.astype(np.float32), sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
        ).T                                                           # (T, 13)

        # Δ / ΔΔ — librosa.feature.delta expects (features, frames) → transpose
        delta  = librosa.feature.delta(mfcc.T, order=1).T            # (T, 13)
        delta2 = librosa.feature.delta(mfcc.T, order=2).T            # (T, 13)

        feats = np.concatenate([mfcc, delta, delta2], axis=-1)       # (T, 39)

        T = feats.shape[0]
        if T >= SEQ_LEN:
            feats = feats[:SEQ_LEN]
        else:
            feats = np.pad(feats, ((0, SEQ_LEN - T), (0, 0)), mode='constant')

        return feats.astype(np.float32)   # (128, 39)

    except Exception as exc:
        print(f"  ⚠  Skipped {file_path}: {exc}")
        return None


def build_dataset(wav_dir: str, labels_csv_path: str,
                  output_name: str = "sep28k_tensors.npz"):
    print("🚀 Starting High-Resolution SEP-28k .wav Processing…")

    df = pd.read_csv(labels_csv_path)

    # ── Scan wav_dir recursively for all .wav files ────────────────────────────
    print(f"🔍 Scanning '{wav_dir}' recursively for .wav files…")
    wav_paths: dict[str, str] = {}
    for root, _, files in os.walk(wav_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_paths[file] = os.path.join(root, file)

    print(f"✅ Found {len(wav_paths)} .wav files. Mapping to CSV labels…")

    if not wav_paths:
        print("❌ No .wav files found! Did you extract the zip archive?")
        return

    # ── Process each labelled row ──────────────────────────────────────────────
    X_list, y_list = [], []
    valid_rows = missing_files = 0

    for _, row in df.iterrows():
        try:
            fname = f"{row['Show']}_{row['EpId']}_{row['ClipId']}.wav"
        except KeyError:
            print("⚠️  CSV columns don't match SEP-28k format. Check your CSV.")
            return

        if fname in wav_paths:
            tensor = process_wav_to_tensor(wav_paths[fname])
            if tensor is not None:
                X_list.append(tensor)
                y_list.append(int(np.argmax(row[TARGET_COLS].values)))
                valid_rows += 1
                if valid_rows % 1000 == 0:
                    print(f"  Processed {valid_rows:,} valid files…")
        else:
            missing_files += 1

    if not X_list:
        print("❌ No tensors created. Processing failed.")
        return

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int32)

    print(f"\n✅ Done!  Tensors: {X.shape}   Labels: {y.shape}")
    print(f"   {missing_files} CSV rows had no matching .wav file.")

    np.savez_compressed(output_name, X=X, y=y)
    print(f"💾 Saved → {output_name}  (ready for training!)")


if __name__ == "__main__":
    WAV_DIRECTORY = "./sep28k_wavs"
    LABELS_CSV    = "SEP-28k_labels.csv"
    build_dataset(WAV_DIRECTORY, LABELS_CSV, output_name="sep28k_tensors.npz")