import librosa
import os
import numpy as np

def extract_mfcc(y, sr, n_mfcc=13):
    """Extract 13-dim mean MFCC."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

folder_path = 'uclass_audio'
filenames = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
failures = 0
for fn in filenames:
    try:
        y, sr = librosa.load(os.path.join(folder_path, fn), sr=16000, duration=3.0)
        feat = extract_mfcc(y, sr)
    except Exception as e:
        print(f"Failed {fn}: {type(e).__name__} - {e}")
        failures += 1
        
print(f"Total failures: {failures}")
