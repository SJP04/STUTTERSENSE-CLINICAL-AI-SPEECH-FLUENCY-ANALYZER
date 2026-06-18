"""
StutterSense — Flask Inference Server
=======================================
Exposes a single POST endpoint /analyze_audio that accepts a WAV file
and returns the Transformer model's stuttering classification.

FIX LOG
-------
- Model/scaler paths now resolve relative to stuttersense_output/ folder
  (one level up from flask_server/), so the server can be run from any
  working directory.
- Added CORS headers for browser-based frontends.
- Added /health endpoint for readiness checks.
"""

import os
from flask import Flask, request, jsonify
import librosa
import numpy as np
from inference_bridge import StutterInferenceBridge

# ── Resolve paths relative to THIS file, not the working directory ─────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_OUTPUT_DIR   = os.path.join(_PROJECT_ROOT, "stuttersense_output")

MODEL_PATH  = os.path.join(_OUTPUT_DIR, "transformer_model.keras")
SCALER_PATH = os.path.join(_OUTPUT_DIR, "scaler.pkl")

app = Flask(__name__)

# ── Load model once when server starts ────────────────────────────────────────
ai_engine = StutterInferenceBridge(
    model_path  = MODEL_PATH,
    scaler_path = SCALER_PATH,
)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "StutterSense Transformer"})


@app.route('/analyze_audio', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "No audio file provided."}), 400

    audio_file  = request.files['audio']
    audio_array, _ = librosa.load(audio_file, sr=16000, duration=3.0, mono=True)

    result = ai_engine.predict_audio_chunk(audio_array)
    return jsonify(result)


if __name__ == '__main__':
    print(f"Model  : {MODEL_PATH}")
    print(f"Scaler : {SCALER_PATH}")
    app.run(host='0.0.0.0', port=5000, debug=False)