import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    pass
import os
import io
import html as html_mod   # for html.escape() in report rendering
import time
import datetime
import wave
import tempfile

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StutterSense – Clinical Detection System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Loader  (reads style.css from the project folder) ─────────────────────
def _load_css(css_file: str = "style.css") -> None:
    """Inject a local CSS file into the Streamlit page."""
    css_path = os.path.join(os.path.dirname(__file__), css_file)
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

_load_css()

# ── Persistent Sticky Header ──────────────────────────────────────────────────
# ── Constants ──────────────────────────────────────────────────────────────────
CLASSES       = ["NoStutter", "Prolongation", "Block", "SoundRepetition", "WordRepetition"]
BADGE_CLASSES = ['badge-nostutter', 'badge-prolongation', 'badge-block', 'badge-soundrep', 'badge-wordrep']
SEP28K_CSV    = os.path.join('sep28k_mfcc', 'sep28k-mfcc.csv')
UCLASS_DIR    = 'uclass_audio'
UCLASS_CSV    = 'uclass_processed.csv'
MFCC_COLS     = [str(i) for i in range(13)]
N_FEATURES    = 39  # 13 MFCC + 13 Delta + 13 Delta-Delta
SEQ_LEN       = 128  # Time-steps per window — MUST match training (test_model.py)

# ── Helper: 39-dim Feature extraction (matches test_model.py exactly) ──────────
HOP_LENGTH = 512  # must match build_wav_dataset.py

def extract_window_features(audio_window: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Convert a raw audio window into a (SEQ_LEN, 39) tensor.
    Uses librosa.feature.delta — exactly matching the training pipeline.
    Returns shape (128, 39).
    """
    mfcc   = librosa.feature.mfcc(y=audio_window, sr=sr, n_mfcc=13,
                                   hop_length=HOP_LENGTH).T            # (T, 13)
    delta  = librosa.feature.delta(mfcc.T, order=1).T                 # (T, 13)
    delta2 = librosa.feature.delta(mfcc.T, order=2).T                 # (T, 13)
    feats  = np.concatenate([mfcc, delta, delta2], axis=-1)           # (T, 39)

    # Pad or truncate to exactly SEQ_LEN frames
    T = feats.shape[0]
    if T >= SEQ_LEN:
        feats = feats[:SEQ_LEN]
    else:
        feats = np.pad(feats, ((0, SEQ_LEN - T), (0, 0)), mode='constant')
    return feats  # (SEQ_LEN, 39)


# ── Legacy helpers kept for UCLASS CSV extraction only ─────────────────────────
def extract_mfcc(y, sr, n_mfcc=13):
    """Extract 13-dim mean MFCC (used only for UCLASS CSV tab)."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# ── Helper: UCLASS folder processing ──────────────────────────────────────────
def extract_uclass_features(folder_path):
    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    if not filenames:
        return None
    all_features, names = [], []
    bar = st.progress(0, text="Extracting MFCC features…")
    for i, fn in enumerate(filenames):
        try:
            y, sr = librosa.load(os.path.join(folder_path, fn), sr=16000, duration=3.0)
            feat = extract_mfcc(y, sr)
            all_features.append(feat)
            names.append(fn)
        except Exception:
            pass
        bar.progress((i + 1) / len(filenames), text=f"Processing {fn}")
    df = pd.DataFrame(all_features, columns=MFCC_COLS)
    df.insert(0, 'filename', names)
    bar.empty()
    return df

# ── Transformer Architecture Components ────────────────────────────────────────
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1.5, alpha=0.25, **kwargs):
        # Pop Keras-internal keys that vary between versions — do NOT forward them.
        _name = kwargs.pop('name', 'focal_loss')
        kwargs.pop('reduction', None)   # discard; let Keras use its own default
        super().__init__(name=_name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.math.log(y_pred)
        one_hot = tf.one_hot(y_true, depth=len(CLASSES))
        p_t = tf.reduce_sum(y_pred * one_hot, axis=-1)
        focal_w = self.alpha * tf.pow(1.0 - p_t, self.gamma)
        loss = focal_w * tf.reduce_sum(ce * one_hot, axis=-1)
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg

def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout=0.1):
    attn_out = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    ffn_out = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ffn_out = tf.keras.layers.Dropout(dropout)(ffn_out)
    ffn_out = tf.keras.layers.Dense(d_model)(ffn_out)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_out)
    return x

def build_model(seq_len=1, n_features=39, n_classes=5):
    d_model = 128
    num_heads = 4
    ff_dim = 256
    dropout = 0.2
    
    inp = tf.keras.Input(shape=(seq_len, n_features))
    x = tf.keras.layers.Dense(d_model)(inp)
    
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = tf.keras.layers.Embedding(input_dim=seq_len, output_dim=d_model)(positions)
    x = x + pos_emb
    x = tf.keras.layers.Dropout(dropout)(x)
    
    # 2 Encoder blocks
    for _ in range(2):
        x = transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inp, outputs=out, name="StutterSense_Transformer")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), 
                  loss=FocalLoss(), metrics=['accuracy'])
    return model

@st.cache_resource(show_spinner=False)
def train_system_model():
    df = pd.read_csv(SEP28K_CSV)
    df = df[
        (df['NoSpeech'] == 0) & 
        (df['DifficultToUnderstand'] == 0) & 
        (df['Unsure'] == 0)
    ].reset_index(drop=True)
    
    mfcc_base = df[MFCC_COLS].values.astype(np.float32)
    # ── Calculate Deltas & Delta-Deltas (39 Features) ────────────────────────
    X = compute_39dim_features(mfcc_base)
    
    tgt_cols = ['NoStutteredWords', 'Prolongation', 'Block', 'SoundRep', 'WordRep']
    y = np.argmax(df[tgt_cols].values, axis=1)
    
    # 1. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 2. Standard Scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. Handle Imbalance with SMOTE
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    except Exception:
        X_train_res, y_train_res = X_train, y_train

    # 4. Data Augmentation: Noise Injection
    noise = np.random.normal(0, 0.05, X_train_res.shape)
    X_train_res = X_train_res + noise
    
    # 5. Reshape for Transformer: (Samples, SeqLen=1, Features=39)
    X_train_res = X_train_res.reshape(-1, 1, 39)
    X_test = X_test.reshape(-1, 1, 39)
    
    model = build_model(seq_len=1, n_features=39)
    
    # 6. Training callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    # 7. Class weights
    counts = np.bincount(y_train_res).astype(float)
    class_wts = {i: counts.sum() / (len(CLASSES) * max(c, 1)) for i, c in enumerate(counts)}
    
    history = model.fit(X_train_res, y_train_res, epochs=50, batch_size=64,
                        validation_data=(X_test, y_test), 
                        class_weight=class_wts,
                        callbacks=[early_stop, lr_reduce],
                        verbose=0)
    
    return model, history, X_test, y_test, scaler

@st.cache_resource(show_spinner=False)
def fetch_trained_model_from_disk():
    """Loads a model and scaler from the stuttersense_output directory. (Forced refresh)"""
    model_path = os.path.join("stuttersense_output", "transformer_model.keras")
    scaler_path = os.path.join("stuttersense_output", "scaler.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None
        
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={"FocalLoss": FocalLoss})
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# ── Audio analysis ─────────────────────────────────────────────────────────────
def analyze_audio(file_path, model, scaler=None, window_sec=3, hop_sec=1, conf_thresh=0.20):
    """
    Slide a window over the audio file and classify each window with the
    Transformer model. Feature pipeline matches test_model.py exactly:
      librosa MFCC (13) → Δ (13) → ΔΔ (13) → pad/truncate (SEQ_LEN, 39) → scale.
    """
    y, sr = librosa.load(file_path, sr=16000)
    ws = int(window_sec * sr)
    hs = int(hop_sec * sr)
    detections, timeline = [], []

    for i in range(0, max(1, len(y) - ws + 1), hs):
        window = y[i: i + ws]

        # ── Step 1: Extract (SEQ_LEN, 39) features ──────────────────────────
        feats = extract_window_features(window, sr)   # (128, 39)

        # ── Step 2: Scale frame-wise (N*T, F) → reshape back → (1, T, F) ──
        if scaler is not None:
            feats = scaler.transform(feats.reshape(-1, N_FEATURES)).reshape(SEQ_LEN, N_FEATURES)
        feat_input = feats.reshape(1, SEQ_LEN, N_FEATURES)            # (1, 128, 39)

        # ── Step 3: Model inference ──────────────────────────────────────────
        prob_vec = model.predict(feat_input, verbose=0)[0]            # (5,)
        idx      = int(np.argmax(prob_vec))
        conf     = float(prob_vec[idx])

        # ── Step 4: UI-Controlled Sensitivity Boost ──────────────────────────
        # Check if the most confident stuttering class exceeds the user's threshold
        stutter_idx = int(np.argmax(prob_vec[1:])) + 1
        stutter_prob = float(prob_vec[stutter_idx])

        if stutter_prob >= conf_thresh:
            idx = stutter_idx
            conf = stutter_prob
        else:
            idx = 0
            conf = float(prob_vec[0])

        # ── Step 5: Silence suppression ──────────────────────────────────────
        # Reject background noise/silence to prevent false "Block" detections
        rms = np.sqrt(np.mean(window ** 2))
        if rms < 0.005:  # Very quiet background
            idx  = 0
            conf = float(prob_vec[0])

        is_stutter = idx > 0
        t_start    = i / sr
        t_end      = min((i + ws) / sr, len(y) / sr)
        probs_map  = {CLASSES[j]: f"{prob_vec[j]*100:.1f}%" for j in range(len(CLASSES))}

        timeline.append({'t_start': t_start, 't_end': t_end,
                         'class_idx': idx, 'class': CLASSES[idx],
                         'conf': conf, 'all_probs': prob_vec})

        if is_stutter:
            mm = int(t_start // 60)
            ss = int(t_start % 60)
            detections.append({
                'Time':       f"{mm:02d}:{ss:02d}",
                'Event':      CLASSES[idx],
                'Confidence': f"{conf*100:.1f}%",
                'Duration':   f"{window_sec}s",
                'Breakdown':  probs_map
            })
    return detections, timeline, y, sr

# ── Plotting helpers ───────────────────────────────────────────────────────────
PALETTE = {
    'NoStutter':        '#8b5cf6',
    'Prolongation':      '#3b82f6',
    'Block':             '#ef4444',
    'SoundRepetition':   '#f59e0b',
    'WordRepetition':    '#10b981',
}

def plot_spectrogram(y, sr, timeline=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), facecolor='#1c2128')
    fig.subplots_adjust(hspace=0.4)
    for ax in (ax1, ax2):
        ax.set_facecolor('#1c2128')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')

    # Mel spectrogram
    S  = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    Sdb = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(Sdb, x_axis='time', y_axis='mel', sr=sr, ax=ax1, cmap='magma')
    ax1.set_title('Mel Spectrogram', color='#a78bfa', fontsize=11, pad=8)

    # Waveform with event overlay
    times = np.linspace(0, len(y)/sr, len(y))
    ax2.plot(times, y, color='#4f46e5', linewidth=0.6, alpha=0.8)
    if timeline:
        for ev in timeline:
            if ev['class_idx'] > 0:
                ax2.axvspan(ev['t_start'], ev['t_end'],
                            alpha=0.25, color=PALETTE.get(ev['class'], 'white'),
                            label=ev['class'])
    ax2.set_title('Waveform + Detected Events', color='#a78bfa', fontsize=11, pad=8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')

    # Legend de-duplicated
    handles = {ev['class']: mpatches.Patch(color=PALETTE.get(ev['class'],'white'), label=ev['class'])
               for ev in (timeline or []) if ev['class_idx'] > 0}
    if handles:
        ax2.legend(handles=handles.values(), loc='upper right',
                   facecolor='#1c2128', edgecolor='#30363d',
                   labelcolor='#e6edf3', fontsize=8)
    plt.tight_layout()
    return fig

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5), facecolor='#1c2128')
    for ax in (ax1, ax2):
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')

    ax1.plot(history.history['accuracy'],     color='#7c3aed', lw=2, label='Train')
    ax1.plot(history.history['val_accuracy'], color='#10b981', lw=2, label='Validation', linestyle='--')
    ax1.set_title('Accuracy', color='#a78bfa'); ax1.legend(labelcolor='#e6edf3', facecolor='#1c2128', edgecolor='#30363d')
    ax1.set_xlabel('Epoch')

    ax2.plot(history.history['loss'],     color='#ef4444', lw=2, label='Train')
    ax2.plot(history.history['val_loss'], color='#f59e0b', lw=2, label='Validation', linestyle='--')
    ax2.set_title('Loss', color='#a78bfa'); ax2.legend(labelcolor='#e6edf3', facecolor='#1c2128', edgecolor='#30363d')
    ax2.set_xlabel('Epoch')

    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#1c2128')
    ax.set_facecolor('#1c2128')
    im = ax.imshow(cm, cmap='Purples', aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels([c.replace(' ','\n') for c in CLASSES], color='#e6edf3', fontsize=8)
    ax.set_yticklabels(CLASSES, color='#e6edf3', fontsize=8)
    ax.set_xlabel('Predicted', color='#8b949e')
    ax.set_ylabel('Actual',    color='#8b949e')
    ax.set_title('Confusion Matrix', color='#a78bfa')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else '#8b949e', fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    plt.tight_layout()
    return fig

def plot_event_distribution(detections):
    counts = {c: 0 for c in CLASSES[1:]}
    for d in detections:
        if d['Event'] in counts:
            counts[d['Event']] += 1
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1c2128')
    ax.set_facecolor('#1c2128')
    bars = ax.barh(list(counts.keys()), list(counts.values()),
                   color=[PALETTE[k] for k in counts], height=0.5, alpha=0.85)
    for bar, val in zip(bars, counts.values()):
        if val > 0:
            ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', color='#e6edf3', fontsize=9)
    ax.set_xlabel('Count', color='#8b949e')
    ax.set_title('Events Detected', color='#a78bfa')
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.tick_params(colors='#8b949e')
    plt.tight_layout()
    return fig

# ── Clinical report generator ──────────────────────────────────────────────────
def generate_report(detections, audio_duration, patient_name="", patient_id="", doctor_name=""):
    now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(detections)
    counts = {c: 0 for c in CLASSES[1:]}
    for d in detections:
        if d['Event'] in counts:
            counts[d['Event']] += 1
    ssi = round(total / max(audio_duration / 60, 0.01), 1)  # events per minute

    if total == 0:
        severity = "None – Fluency within normal limits"
    elif ssi < 5:
        severity = "Mild Stuttering"
    elif ssi < 15:
        severity = "Moderate Stuttering"
    else:
        severity = "Severe Stuttering"

    dominant = max(counts, key=counts.get) if total > 0 else "N/A"

    report = f"""
{"=" * 68}
{"STUTTER SENSE  -  CLINICAL SPEECH ANALYSIS REPORT".center(68)}
{"=" * 68}

  Patient Name  : {patient_name or '—'}
  Patient ID    : {patient_id   or '—'}
  Report Date   : {now}
  Diagnosing Dr : {doctor_name or '—'}
  Dataset Used  : SEP-28k + UCLASS Archive

{"-" * 68}
  RECORDING SUMMARY
{"-" * 68}
  Audio Duration       : {audio_duration:.1f} seconds ({audio_duration/60:.1f} min)
  Total Events Flagged : {total}
  Stutter Rate (SSI)   : {ssi} events/minute
  Severity Assessment  : {severity}

{"-" * 68}
  EVENT BREAKDOWN
{"-" * 68}
  Prolongation     (sustained sounds)  : {counts['Prolongation']:>4}
  Block            (silent stoppages)  : {counts['Block']:>4}
  SoundRepetition  (phoneme repeats)   : {counts['SoundRepetition']:>4}
  WordRepetition   (word-level reps)   : {counts['WordRepetition']:>4}
  {"-" * 34}
  Dominant Stutter Type                : {dominant}

{"-" * 68}
  DETAILED TIMESTAMPS
{"-" * 68}"""
    if detections:
        for d in detections:
            report += f"\n  [{d['Time']}]  {d['Event']:<20}  Confidence: {d['Confidence']}"
    else:
        report += "\n  No stuttering events detected."

    report += f"""

{"-" * 68}
  CLINICAL RECOMMENDATIONS
{"-" * 68}"""
    if total == 0:
        report += "\n  ✔ Speech fluency is within normal limits.\n  ✔ No intervention required at this time."
    elif ssi < 5:
        report += "\n  • Mild disfluency detected. Monitor over subsequent sessions.\n  • Consider relaxation and breath-control exercises."
    elif ssi < 15:
        report += "\n  • Moderate stuttering. Recommend structured therapy programme.\n  • Focus on smooth speech / easy-onset techniques.\n  • Re-evaluate in 4–6 weeks."
    else:
        report += "\n  ⚠ Severe stuttering. Immediate referral to a certified SLP recommended.\n  • Intensive therapy (e.g. Lidcombe / SpeakMore Fluently) advised.\n  • Monitor emotional/psychological impact."

    report += f"\n\n  {'-' * 61}\n"
    report += "  This report is generated by an AI system and must be reviewed\n"
    report += "  by a qualified Speech-Language Pathologist before clinical use.\n"
    report += f"  {'-' * 61}\n"
    return report

def generate_html_preview(detections, audio_duration, patient_name="", patient_id="", doctor_name=""):
    now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(detections)
    counts = {c: 0 for c in CLASSES[1:]}
    for d in detections:
        if d['Event'] in counts:
            counts[d['Event']] += 1
    ssi = round(total / max(audio_duration / 60, 0.01), 1)

    if total == 0:
        severity = "Normal Fluency"
        rec = "✔ Speech fluency is within normal limits.<br>✔ No intervention required at this time."
    elif ssi < 5:
        severity = "Mild"
        rec = "• Mild disfluency detected. Monitor over subsequent sessions.<br>• Consider relaxation exercises."
    elif ssi < 15:
        severity = "Moderate"
        rec = "• Moderate stuttering. Recommend structured therapy programme.<br>• Re-evaluate in 4–6 weeks."
    else:
        severity = "Severe"
        rec = "⚠ Severe stuttering. Immediate referral to a certified SLP recommended.<br>• Intensive therapy advised."

    dominant = max(counts, key=counts.get) if total > 0 else "N/A"
    
    html = f"""
<div style="font-family: 'Inter', sans-serif; color: #e6edf3; background: #1c2128; padding: 2rem; border-radius: 12px; border: 1px solid #30363d; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
    <div style="text-align: center; border-bottom: 1px solid #30363d; padding-bottom: 1.5rem; margin-bottom: 1.5rem;">
        <h2 style="margin: 0; color: #a78bfa; font-weight: 700; letter-spacing: 1.5px;">STUTTER SENSE</h2>
        <p style="margin: 0.5rem 0 0 0; color: #8b949e; letter-spacing: 2px; font-size: 0.85rem;">CLINICAL SPEECH ANALYSIS REPORT</p>
    </div>
    
    <div style="display: flex; justify-content: space-between; margin-bottom: 2.5rem; font-size: 0.95rem;">
        <div>
            <p style="margin: 0 0 0.5rem 0;"><span style="color: #8b949e">Patient Name :</span> <b style="color: white; margin-left: 5px;">{patient_name or '—'}</b></p>
            <p style="margin: 0;"><span style="color: #8b949e">Patient ID   :</span> <b style="color: white; margin-left: 5px;">{patient_id or '—'}</b></p>
        </div>
        <div style="text-align: right;">
            <p style="margin: 0 0 0.5rem 0;"><span style="color: #8b949e">Report Date :</span> <b style="color: white; margin-left: 5px;">{now}</b></p>
            <p style="margin: 0;"><span style="color: #8b949e">Diagnosing Dr:</span> <span style="margin-left: 5px; color: white; font-weight: bold;">{doctor_name or '—'}</span></p>
        </div>
    </div>

    <h4 style="color: #a78bfa; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; font-size: 0.9rem; margin-bottom: 1rem;">RECORDING SUMMARY</h4>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2.5rem;">
        <div style="background: #161b22; padding: 1.2rem; border-radius: 8px; border: 1px solid #30363d; text-align: center;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #3b82f6;">{audio_duration:.1f}s</div>
            <div style="font-size: 0.75rem; color: #8b949e; text-transform: uppercase; margin-top: 5px; font-weight: 600;">Audio Duration</div>
        </div>
        <div style="background: #161b22; padding: 1.2rem; border-radius: 8px; border: 1px solid #30363d; text-align: center;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #f59e0b;">{total}</div>
            <div style="font-size: 0.75rem; color: #8b949e; text-transform: uppercase; margin-top: 5px; font-weight: 600;">Events Flagged</div>
        </div>
        <div style="background: #161b22; padding: 1.2rem; border-radius: 8px; border: 1px solid #30363d; text-align: center;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #ef4444;">{ssi}</div>
            <div style="font-size: 0.75rem; color: #8b949e; text-transform: uppercase; margin-top: 5px; font-weight: 600;">Stutter Rate (/min)</div>
        </div>
        <div style="background: #161b22; padding: 1.2rem; border-radius: 8px; border: 1px solid #30363d; text-align: center;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #10b981;">{severity}</div>
            <div style="font-size: 0.75rem; color: #8b949e; text-transform: uppercase; margin-top: 5px; font-weight: 600;">Severity</div>
        </div>
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
        <div>
            <h4 style="color: #a78bfa; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; font-size: 0.9rem; margin-bottom: 1rem;">EVENT BREAKDOWN</h4>
            <div style="background: #161b22; border-radius: 8px; border: 1px solid #30363d; padding: 1rem;">
                <table style="width: 100%; font-size: 0.95rem; border-collapse: collapse; table-layout: fixed;">
                    <tr style="border-bottom: 1px solid #30363d;">
                        <td style="padding: 0.8rem 0.5rem; width: 40%;">Prolongation</td>
                        <td style="color: #8b949e; font-size: 0.8rem; padding: 0.8rem 0.5rem; width: 45%;">(sustained sounds)</td>
                        <td style="color: #3b82f6; text-align: right; font-weight: bold; padding: 0.8rem 0.5rem; width: 15%;">{counts['Prolongation']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #30363d;">
                        <td style="padding: 0.8rem 0.5rem;">Block</td>
                        <td style="color: #8b949e; font-size: 0.8rem; padding: 0.8rem 0.5rem;">(silent stoppages)</td>
                        <td style="color: #ef4444; text-align: right; font-weight: bold; padding: 0.8rem 0.5rem;">{counts['Block']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #30363d;">
                        <td style="padding: 0.8rem 0.5rem;">SoundRepetition</td>
                        <td style="color: #8b949e; font-size: 0.8rem; padding: 0.8rem 0.5rem;">(phoneme repeats)</td>
                        <td style="color: #f59e0b; text-align: right; font-weight: bold; padding: 0.8rem 0.5rem;">{counts['SoundRepetition']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #30363d;">
                        <td style="padding: 0.8rem 0.5rem;">WordRepetition</td>
                        <td style="color: #8b949e; font-size: 0.8rem; padding: 0.8rem 0.5rem;">(word-level reps)</td>
                        <td style="color: #10b981; text-align: right; font-weight: bold; padding: 0.8rem 0.5rem;">{counts['WordRepetition']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.8rem 0.5rem; color: #a78bfa;">Dominant Type</td>
                        <td style="padding: 0.8rem 0.5rem;"></td>
                        <td style="text-align: right; font-weight: bold; padding: 0.8rem 0.5rem; color: white;">{dominant}</td>
                    </tr>
                </table>
            </div>
        </div>
        <div>
            <h4 style="color: #a78bfa; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; font-size: 0.9rem; margin-bottom: 1rem;">CLINICAL RECOMMENDATIONS</h4>
            <div style="background: rgba(124, 58, 237, 0.05); border-left: 3px solid #7c3aed; padding: 1.2rem; border-radius: 4px; font-size: 0.9rem; line-height: 1.6;">
                {rec}
            </div>
        </div>
    </div>
</div>
"""
    # Force single-line raw HTML to bypass Streamlit's aggressive Markdown codeblock parser
    return "".join(line.strip() for line in html.split("\n"))

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div class="sidebar-logo">
  <h1>🎙️ StutterSense</h1>
  <p>CLINICAL AI PLATFORM</p>
</div>
""", unsafe_allow_html=True)

ALL_PAGES = ["🏠 Home", "📊 Dataset & Training", "🔬 Audio Analysis", "🎤 Live Recording", "📋 Clinical Report"]

# ── Nav-card redirect: write directly into the radio widget's session key ──
# This is the only reliable way to programmatically change a keyed radio button.
if "_nav_target" in st.session_state:
    _target = st.session_state.pop("_nav_target")
    if _target in ALL_PAGES:
        st.session_state["_sidebar_nav"] = _target

# Initialise key so radio renders correctly on first load
if "_sidebar_nav" not in st.session_state:
    st.session_state["_sidebar_nav"] = "🏠 Home"

page = st.sidebar.radio(
    "Navigation",
    ALL_PAGES,
    label_visibility="collapsed",
    key="_sidebar_nav",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")

model_ready = 'trained_model' in st.session_state
st.sidebar.markdown(
    f"{'✅ Model Ready' if model_ready else '⚠️ Model not trained'}",
    unsafe_allow_html=True,
)

if os.path.exists(SEP28K_CSV):
    n = len(pd.read_csv(SEP28K_CSV, usecols=['PoorAudioQuality']))
    st.sidebar.markdown(f"📂 SEP-28k: **{n:,} clips**")
else:
    st.sidebar.markdown("❌ SEP-28k CSV missing")

n_uclass = len(os.listdir(UCLASS_DIR)) if os.path.exists(UCLASS_DIR) else 0
st.sidebar.markdown(f"📂 UCLASS: **{n_uclass} files**")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#8b949e'>v2.0 · Built with Streamlit + TF · SEP-28k</small>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":

    # ── Hero Banner ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-orb hero-orb-1"></div>
      <div class="hero-orb hero-orb-2"></div>
      <div class="hero-orb hero-orb-3"></div>
      <div class="hero-content">
        <div class="hero-badge">🧠 AI · Speech Pathology · Deep Learning</div>
        <h1 class="hero-title">StutterSense</h1>
        <p class="hero-subtitle">Clinical Stuttering Detection System</p>
        <p class="hero-desc">
          Automated dysfluency analysis powered by CNN-BiLSTM neural networks &amp;
          MFCC signal processing — built for speech-language pathologists.
        </p>
        <div class="hero-pills">
          <span class="hero-pill">🎙️ SEP-28k Dataset</span>
          <span class="hero-pill">📡 UCLASS Archive</span>
          <span class="hero-pill">⚡ Real-time Analysis</span>
          <span class="hero-pill">📋 Clinical Reports</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Live Status Bar ────────────────────────────────────────────────────────
    model_status  = "✅ Ready" if 'trained_model' in st.session_state else "⚠️ Not Trained"
    model_color   = "#10b981"  if 'trained_model' in st.session_state else "#f59e0b"
    sep28k_status = "✅ Loaded" if os.path.exists(SEP28K_CSV) else "❌ Missing"
    sep28k_color  = "#10b981"  if os.path.exists(SEP28K_CSV) else "#ef4444"
    n_uclass_files = len(os.listdir(UCLASS_DIR)) if os.path.exists(UCLASS_DIR) else 0

    st.markdown(f"""
    <div class="status-bar">
      <div class="status-item">
        <span class="status-dot" style="background:{model_color}"></span>
        <span class="status-label">AI Model</span>
        <span class="status-value" style="color:{model_color}">{model_status}</span>
      </div>
      <div class="status-divider"></div>
      <div class="status-item">
        <span class="status-dot" style="background:{sep28k_color}"></span>
        <span class="status-label">SEP-28k Dataset</span>
        <span class="status-value" style="color:{sep28k_color}">{sep28k_status}</span>
      </div>
      <div class="status-divider"></div>
      <div class="status-item">
        <span class="status-dot" style="background:#3b82f6"></span>
        <span class="status-label">UCLASS Files</span>
        <span class="status-value" style="color:#60a5fa">{n_uclass_files} recordings</span>
      </div>
      <div class="status-divider"></div>
      <div class="status-item">
        <span class="status-dot" style="background:#a78bfa; animation: pulse-dot 1.5s infinite;"></span>
        <span class="status-label">System</span>
        <span class="status-value" style="color:#a78bfa">Online</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation Cards ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:2rem">🚀 Navigate to a Module</div>',
                unsafe_allow_html=True)

    nav_data = [
        ("📊", "Dataset & Training",
         "Process UCLASS audio and train the CNN-BiLSTM model on SEP-28k features.",
         ["MFCC Extraction", "CNN-BiLSTM Training", "Confusion Matrix"],
         "#7c3aed", "#4f46e5", "📊 Dataset & Training"),
        ("🔬", "Audio Analysis",
         "Upload any .wav file and get timestamped stuttering event detection.",
         ["Event Timestamps", "Mel Spectrogram", "Waveform Overlay"],
         "#3b82f6", "#1d4ed8", "🔬 Audio Analysis"),
        ("🎤", "Live Recording",
         "Record your voice directly and analyse dysfluency in real time.",
         ["Mic Input", "Instant Detection", "Live Spectrogram"],
         "#10b981", "#059669", "🎤 Live Recording"),
        ("📋", "Clinical Report",
         "Generate a downloadable diagnostic report for the speech pathologist.",
         ["Severity Score", "SSI Rate", "Download .txt"],
         "#f59e0b", "#d97706", "📋 Clinical Report"),
    ]

    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    for col, (icon, title, desc, tags, c1, c2, target_page) in zip(cols, nav_data):
        with col:
            # Render the visual card
            tag_html = "".join(f'<span class="nav-tag">{t}</span>' for t in tags)
            st.markdown(f"""
            <div class="nav-card" style="--card-c1:{c1};--card-c2:{c2};">
              <div class="nav-card-glow"></div>
              <div class="nav-icon">{icon}</div>
              <div class="nav-title">{title}</div>
              <div class="nav-desc">{desc}</div>
              <div class="nav-tags">{tag_html}</div>
              <div class="nav-arrow">→</div>
            </div>
            """, unsafe_allow_html=True)
            # Actual clickable Streamlit button below the card
            if st.button(f"Open {title}", key=f"nav_{title}", use_container_width=True):
                st.session_state["_nav_target"] = target_page
                st.rerun()

    # ── Stats Row ──────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    stat_cards = [
        ("5",    "Stutter Classes",    "📌", "#a78bfa"),
        ("39",   "Hybrid Features",    "🔊", "#60a5fa"),
        ("138",  "UCLASS Recordings",  "📁", "#34d399"),
        ("Transformer", "Neural Architecture","🧠", "#fbbf24"),
    ]
    for col, (val, lbl, ic, clr) in zip([sc1, sc2, sc3, sc4], stat_cards):
        col.markdown(f"""
        <div class="metric-card" style="text-align:center;border-top:3px solid {clr};">
          <div style="font-size:1.6rem">{ic}</div>
          <div class="metric-value" style="color:{clr};font-size:1.7rem">{val}</div>
          <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    # ── Architecture + Stutter Types ──────────────────────────────────────────
    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown(
"""
<style>
.acoustic-card {
background: linear-gradient(145deg, #161b22 0%, #1c2128 100%);
border: 1px solid #30363d;
border-radius: 16px;
padding: 2rem;
position: relative;
overflow: hidden;
box-shadow: 0 10px 30px rgba(0,0,0,0.2);
margin-top: 0.5rem;
}
.acoustic-card::before {
content: '';
position: absolute;
top: 0; left: 0; right: 0;
height: 4px;
background: linear-gradient(90deg, #7c3aed, #3b82f6);
}
.audio-visualizer {
display: flex;
justify-content: center;
align-items: center;
gap: 6px;
margin: 2.5rem 0;
height: 60px;
}
.audio-bar {
width: 8px;
border-radius: 4px;
background: #a78bfa;
animation: eq-bounce 1s ease-in-out infinite;
animation-delay: var(--delay);
}
@keyframes eq-bounce {
0%, 100% { height: 15px; opacity: 0.6; }
50% { height: 60px; opacity: 1; background: #7c3aed; }
}
.engine-stats {
display: flex;
justify-content: space-between;
background: #0d1117;
border-radius: 12px;
padding: 1.2rem;
border: 1px solid #30363d;
}
.e-stat { text-align: center; width: 33%; }
.e-val { font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.e-lbl { font-size: 0.75rem; color: #8b949e; letter-spacing: 1px; margin-top: 0.2rem; }
</style>

<div class="acoustic-card">
<h3 style="margin:0 0 0.5rem 0; color:#e6edf3; font-weight:600;">Clinical Dysfluency Engine</h3>
<p style="margin:0; color:#8b949e; font-size:0.95rem; line-height:1.5;">
The AI module continuously analyzes acoustic properties, identifying subtle stuttering phenotypes entirely objective to support formal clinical evaluation.
</p>

<div class="audio-visualizer">
<div class="audio-bar" style="--delay: 0.0s;"></div>
<div class="audio-bar" style="--delay: 0.3s;"></div>
<div class="audio-bar" style="--delay: 0.1s;"></div>
<div class="audio-bar" style="--delay: 0.4s;"></div>
<div class="audio-bar" style="--delay: 0.2s;"></div>
<div class="audio-bar" style="--delay: 0.5s;"></div>
<div class="audio-bar" style="--delay: 0.1s;"></div>
<div class="audio-bar" style="--delay: 0.3s;"></div>
<div class="audio-bar" style="--delay: 0.0s;"></div>
<div class="audio-bar" style="--delay: 0.4s;"></div>
<div class="audio-bar" style="--delay: 0.2s;"></div>
</div>

<div class="engine-stats">
<div class="e-stat">
<div class="e-val" style="color:#a78bfa;">16kHz</div>
<div class="e-lbl">SAMPLING RATE</div>
</div>
<div style="width:1px; background:#30363d;"></div>
<div class="e-stat">
<div class="e-val" style="color:#10b981;">MULTI</div>
<div class="e-lbl">CLASS DETECTION</div>
</div>
<div style="width:1px; background:#30363d;"></div>
<div class="e-stat">
<div class="e-val" style="color:#3b82f6;">OBJECTIVE</div>
<div class="e-lbl">CLINICAL METRICS</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)




    with col_r:
        st.markdown('<div class="section-title">🔖 Stutter Event Types</div>', unsafe_allow_html=True)
        events_info = {
            "Prolongation":     ("🔵", "#3b82f6", "Stretching of a sound (e.g. 'sssseee')"),
            "Block":            ("🔴", "#ef4444", "Complete airflow stoppage mid-speech"),
            "SoundRepetition":  ("🟡", "#f59e0b", "Phoneme repeated (e.g. 'b-b-ball')"),
            "WordRepetition":   ("🟢", "#10b981", "Whole word repeated (e.g. 'I-I-I want')"),
        }
        for ev, (ic, clr, desc) in events_info.items():
            st.markdown(f"""
            <div class="metric-card" style="padding:0.85rem 1.1rem;margin-bottom:0.55rem;
                         border-left:3px solid {clr};">
              <b style="color:{clr}">{ic} {ev}</b><br>
              <small style="color:#8b949e">{desc}</small>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DATASET & TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset & Training":
    st.markdown('<h2 class="section-title">📊 Dataset Processing & Model Training</h2>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🗂️ UCLASS Audio Processing", "🧠 Transformer Model Training"])

    with tab1:
        st.markdown("**Extract MFCC features from the UCLASS `.wav` recordings stored in `uclass_audio/`.**")
        col_info, col_btn = st.columns([3,1])
        with col_info:
            n_wav = len([f for f in os.listdir(UCLASS_DIR) if f.lower().endswith('.wav')]) if os.path.exists(UCLASS_DIR) else 0
            st.info(f"📁 Found **{n_wav} .wav files** in `{UCLASS_DIR}/`")
        clicked_extract = False
        with col_btn:
            clicked_extract = st.button("▶ Extract Features", use_container_width=True)
            
        if clicked_extract:
            if n_wav == 0:
                st.error("No .wav files found in uclass_audio/")
            else:
                with st.spinner("Processing UCLASS audio files…"):
                    udf = extract_uclass_features(UCLASS_DIR)
                if udf is not None:
                    udf.to_csv(UCLASS_CSV, index=False)
                    st.success(f"✅ Extracted {len(udf)} feature vectors → saved to `{UCLASS_CSV}`")
                    st.markdown("<br>**Currently processed UCLASS features:**", unsafe_allow_html=True)
                    udf_cur = udf.copy()
                    mfcc_rename = {str(i): f"MFCC-{i}" for i in range(13)}
                    if all(str(i) in udf_cur.columns for i in range(13)):
                        udf_cur = udf_cur.rename(columns=mfcc_rename)
                    num_cols = [c for c in udf_cur.columns if c.startswith("MFCC")]
                    if num_cols:
                        udf_cur[num_cols] = udf_cur[num_cols].round(2)
                    st.dataframe(udf_cur, use_container_width=True, hide_index=False)

        if os.path.exists(UCLASS_CSV):
            st.markdown("---")
            st.markdown("**Previously processed UCLASS features:**")
            udf_prev = pd.read_csv(UCLASS_CSV)
            # Rename MFCC columns for readability
            mfcc_rename = {str(i): f"MFCC-{i}" for i in range(13)}
            udf_display = udf_prev.rename(columns=mfcc_rename)
            # Round numeric columns to 2 decimal places
            num_cols = [c for c in udf_display.columns if c.startswith("MFCC")]
            udf_display[num_cols] = udf_display[num_cols].round(2)
            st.dataframe(
                udf_display,
                use_container_width=True,
                hide_index=False,
            )
            st.markdown(
                f"<small style='color:#8b949e'>📄 {len(udf_prev)} recordings &nbsp;·&nbsp; "
                f"{len(udf_prev.columns)} columns (filename + 13 MFCCs) &nbsp;·&nbsp; "
                f"saved to <code>uclass_processed.csv</code></small>",
                unsafe_allow_html=True,
            )

    with tab2:
        st.markdown("**Train the Transformer model on SEP-28k MFCC features (robust noise-inclusive training).**")

        col_a, col_b = st.columns([1,1])
        with col_a:
            if not os.path.exists(SEP28K_CSV):
                st.error(f"`{SEP28K_CSV}` not found. Place it in the project folder.")
            else:
                df_info = pd.read_csv(SEP28K_CSV)
                good    = df_info[
                    (df_info['NoSpeech'] == 0) & 
                    (df_info['DifficultToUnderstand'] == 0) & 
                    (df_info['Unsure'] == 0)
                ]
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{len(good):,}</div>
                  <div class="metric-label">Qualified training clips</div>
                </div>""", unsafe_allow_html=True)

        with col_b:
            # Check if a saved model exists on disk
            output_dir = "stuttersense_output"
            saved_model_exists = os.path.exists(os.path.join(output_dir, "transformer_model.keras"))
            
            if saved_model_exists:
                if st.button("📂 Load Pre-trained Model", use_container_width=True):
                    with st.spinner("Loading saved model from disk..."):
                        model, scaler = fetch_trained_model_from_disk()
                        if model and scaler:
                            st.session_state['trained_model'] = model
                            st.session_state['fitted_scaler'] = scaler
                            st.success("✅ Model and Scaler loaded successfully from `stuttersense_output/`!")
                            st.rerun()
            else:
                st.warning("⚠️ No pre-trained model found in `stuttersense_output/`. Please run `test_model.py` in your terminal first.")

        if 'train_history' in st.session_state:
            hist = st.session_state['train_history']
            final_acc = hist.history['val_accuracy'][-1]
            final_loss = hist.history['val_loss'][-1]

            mc1, mc2 = st.columns(2)
            mc1.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{final_acc*100:.1f}%</div>
              <div class="metric-label">Validation Accuracy</div>
            </div>""", unsafe_allow_html=True)
            mc2.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{final_loss:.4f}</div>
              <div class="metric-label">Validation Loss</div>
            </div>""", unsafe_allow_html=True)

            st.pyplot(plot_training_history(hist), use_container_width=True)
        elif 'trained_model' in st.session_state and os.path.exists(os.path.join("stuttersense_output", "training_curves.png")):
            # Display pre-trained metrics if available
            report_path = os.path.join("stuttersense_output", "classification_report.txt")
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    lines = f.readlines()
                    # Try to find accuracy in the report
                    acc_line = [l for l in lines if "accuracy" in l.lower() and "avg" not in l.lower()]
                    if acc_line:
                        parts = acc_line[0].split()
                        acc_val = parts[-2] # Usually the penultimate item
                        st.markdown(f"""
                        <div class="metric-card" style="border-top:3px solid #10b981">
                          <div class="metric-value" style="color:#10b981">{float(acc_val)*100:.1f}%</div>
                          <div class="metric-label">Pre-trained Validation Accuracy</div>
                        </div>""", unsafe_allow_html=True)

            st.image(os.path.join("stuttersense_output", "training_curves.png"), caption="Training History (Loaded from disk)", use_container_width=True)

        if 'trained_model' in st.session_state:
            # Confusion matrix / Report logic
            output_dir = "stuttersense_output"
            cm_path = os.path.join(output_dir, "confusion_matrix.png")
            report_path = os.path.join(output_dir, "classification_report.txt")

            cmt1, cmt2 = st.columns([1,1])
            with cmt1:
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
                else:
                    # Fallback to generating it if X_test/y_test exists
                    if 'X_test' in st.session_state:
                        model = st.session_state['trained_model']
                        X_test = st.session_state['X_test']
                        y_test = st.session_state['y_test']
                        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                        st.pyplot(plot_confusion_matrix(y_test, y_pred), use_container_width=True)
            with cmt2:
                if os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        report_str = f.read()
                    report_safe = html_mod.escape(report_str).replace(' ', '&nbsp;').replace('\n', '<br>')
                    st.markdown(
                        f'<div class="report-box" style="font-family: \'JetBrains Mono\', monospace;">{report_safe}</div>',
                        unsafe_allow_html=True,
                    )
                elif 'X_test' in st.session_state:
                    model = st.session_state['trained_model']
                    X_test = st.session_state['X_test']
                    y_test = st.session_state['y_test']
                    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                    report_str = classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0)
                    report_safe = html_mod.escape(report_str).replace(' ', '&nbsp;').replace('\n', '<br>')
                    st.markdown(
                        f'<div class="report-box" style="font-family: \'JetBrains Mono\', monospace;">{report_safe}</div>',
                        unsafe_allow_html=True,
                    )

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: AUDIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Audio Analysis":
    st.markdown('<h2 class="section-title">🔬 Audio File Analysis</h2>', unsafe_allow_html=True)

    if 'trained_model' not in st.session_state:
        st.warning("⚠️ Please train the model first on the **Dataset & Training** page.")
        st.stop()

    uploaded = st.file_uploader("Upload a `.wav` file for diagnosis", type=["wav"])
    if not uploaded:
        st.info("👆 Upload a WAV recording to begin analysis.")
        st.stop()

    tmp_path = os.path.join(tempfile.gettempdir(), "stutter_upload.wav")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    col_settings, col_run = st.columns([3,1])
    with col_settings:
        window_sec = st.slider("Analysis window (seconds)", 1, 5, 3)
        # Lowered threshold slider to match the natural probability distributions
        conf_thresh = st.slider("Confidence threshold", 0.0, 0.50, 0.15, 0.01)
    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🔍 Analyse Audio", use_container_width=True)

    if run_btn:
        with st.spinner("Analysing speech patterns…"):
            detections, timeline, audio_y, sr = analyze_audio(
                tmp_path, st.session_state['trained_model'],
                scaler=st.session_state.get('fitted_scaler'),
                window_sec=window_sec, hop_sec=1, conf_thresh=conf_thresh
            )
        st.session_state['last_detections']  = detections
        st.session_state['last_timeline']    = timeline
        st.session_state['last_audio_y']     = audio_y
        st.session_state['last_sr']          = sr
        st.session_state['last_duration']    = len(audio_y) / sr

    if 'last_detections' in st.session_state:
        detections = st.session_state['last_detections']
        timeline   = st.session_state['last_timeline']
        audio_y    = st.session_state['last_audio_y']
        sr         = st.session_state['last_sr']
        duration   = st.session_state['last_duration']

        # Top metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        events_count = len(detections)
        ssi = round(events_count / max(duration/60, 0.01), 1)
        dominant = max({c:sum(1 for d in detections if d['Event']==c) for c in CLASSES[1:]},
                       key=lambda k: sum(1 for d in detections if d['Event']==k)) if detections else "None"
        severity_label = "None" if events_count==0 else "Mild" if ssi<5 else "Moderate" if ssi<15 else "Severe"

        for col, val, lbl in zip(
            [mc1,mc2,mc3,mc4],
            [events_count, f"{ssi}/min", dominant, severity_label],
            ["Events Detected","Stutter Rate","Dominant Type","Severity"],
        ):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="font-size:1.5rem">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        # Spectrogram & Probability Heatmap
        st.pyplot(plot_spectrogram(audio_y, sr, timeline), use_container_width=True)
        
        # Heatmap of all stutter classes
        st.markdown("### 📊 Multi-Class Confidence Breakdown")
        if timeline:
            time_axis = [t['t_start'] for t in timeline]
            prob_matrix = np.array([t['all_probs'][1:] for t in timeline]).T # Exclude NoStutter
            fig, ax = plt.subplots(figsize=(12, 2.5))
            im = ax.imshow(prob_matrix, aspect='auto', cmap='magma', extent=[0, max(time_axis), 0, 4])
            ax.set_yticks(range(4))
            ax.set_yticklabels(CLASSES[1:][::-1])
            ax.set_xlabel("Time (s)")
            plt.colorbar(im, label="Confidence")
            st.pyplot(fig, use_container_width=True)

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown("### Detection Timestamps & Confidence Breakdown")
            if detections:
                df_det = pd.DataFrame(detections)
                # Flatten the breakdown for display
                for cls in CLASSES:
                    df_det[f"Prob_{cls}"] = df_det['Breakdown'].apply(lambda x: x.get(cls, "0%"))
                
                display_cols = ['Time', 'Event', 'Confidence'] + [f"Prob_{cls}" for cls in CLASSES[1:]]
                st.dataframe(df_det[display_cols], use_container_width=True, hide_index=True)
            else:
                st.success("✅ No significant stuttering events detected.")

        with col_right:
            st.markdown("### Event Distribution")
            if detections:
                st.pyplot(plot_event_distribution(detections), use_container_width=True)
            else:
                st.info("No events to display.")

        st.markdown("### 🔊 Original Audio")
        st.audio(uploaded)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LIVE RECORDING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎤 Live Recording":
    st.markdown('<h2 class="section-title">🎤 Live Voice Input</h2>', unsafe_allow_html=True)

    if 'trained_model' not in st.session_state:
        st.warning("⚠️ Please train the model first on the **Dataset & Training** page.")
        st.stop()

    st.markdown("""
    <div class="metric-card" style="border-left:4px solid #10b981;padding:1rem 1.4rem">
      <b style="color:#34d399">📌 Instructions</b><br>
      <small style="color:#8b949e">
        Use the recorder below to capture speech directly from your microphone.
        When done, click <b>Analyse</b> to detect stuttering events in real time.
      </small>
    </div>
    """, unsafe_allow_html=True)

    audio_bytes = st.audio_input("🎙️ Click to record your voice", key="live_audio")

    if audio_bytes:
        tmp_path = os.path.join(tempfile.gettempdir(), "live_recording.wav")
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes.getbuffer())

        col_a, col_b = st.columns([1,1])
        with col_a:
            st.markdown("**Recorded audio preview:**")
            st.audio(audio_bytes)
        with col_b:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ Analyse Live Recording", use_container_width=True):
                with st.spinner("Analysing live speech…"):
                    try:
                        detections, timeline, audio_y, sr = analyze_audio(
                            tmp_path, st.session_state['trained_model'],
                            scaler=st.session_state.get('fitted_scaler')
                        )
                        st.session_state['live_detections'] = detections
                        st.session_state['live_timeline']   = timeline
                        st.session_state['live_audio_y']   = audio_y
                        st.session_state['live_sr']        = sr
                        st.session_state['live_duration']  = len(audio_y)/sr
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"Could not process audio: {e}")

    if 'live_detections' in st.session_state:
        detections = st.session_state['live_detections']
        timeline   = st.session_state['live_timeline']
        audio_y    = st.session_state['live_audio_y']
        sr         = st.session_state['live_sr']
        duration   = st.session_state['live_duration']

        mc1, mc2, mc3 = st.columns(3)
        events_count  = len(detections)
        ssi           = round(events_count / max(duration/60, 0.01), 1)
        severity_label= "None" if events_count==0 else "Mild" if ssi<5 else "Moderate" if ssi<15 else "Severe"

        for col, val, lbl in zip(
            [mc1,mc2,mc3],
            [events_count, f"{ssi}/min", severity_label],
            ["Events Found","Stutter Rate","Severity"],
        ):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="font-size:1.5rem">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.pyplot(plot_spectrogram(audio_y, sr, timeline), use_container_width=True)

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown("### 📋 Detection Timestamps & Breakdown")
            if detections:
                df_det = pd.DataFrame(detections)
                for cls in CLASSES:
                    df_det[f"P({cls})"] = df_det['Breakdown'].apply(lambda x: x.get(cls, "0%"))
                
                display_cols = ['Time', 'Event', 'Confidence'] + [f"P({cls})" for cls in CLASSES[1:]]
                st.dataframe(df_det[display_cols], use_container_width=True, hide_index=True)
            else:
                st.success("✅ Speech fluency within normal limits – no stuttering detected.")

        with col_right:
            st.markdown("### Event Distribution")
            if detections:
                st.pyplot(plot_event_distribution(detections), use_container_width=True)
            else:
                st.info("No events to display.")

        st.markdown("---")
        st.markdown("**Want a full clinical report?** Navigate to **📋 Clinical Report** in the sidebar.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CLINICAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Clinical Report":
    st.markdown('<h2 class="section-title">📋 Clinical Report Generator</h2>', unsafe_allow_html=True)

    # Decide source
    source = st.radio(
        "Select analysis source:",
        ["Uploaded Audio Analysis", "Live Recording Analysis"],
        horizontal=True,
    )

    det_key  = 'last_detections'  if source.startswith("Uploaded") else 'live_detections'
    dur_key  = 'last_duration'    if source.startswith("Uploaded") else 'live_duration'

    if det_key not in st.session_state:
        st.info("ℹ️ Run an analysis first (Audio Analysis or Live Recording page).")
        st.stop()

    detections = st.session_state[det_key]
    duration   = st.session_state[dur_key]

    st.markdown("### 👤 Patient & Clinician Information")
    p1, p2, p3 = st.columns(3)
    patient_name = p1.text_input("Patient Name", placeholder="e.g. John Doe")
    patient_id   = p2.text_input("Patient ID",   placeholder="e.g. P-20240001")
    doctor_name  = p3.text_input("Doctor Name",  placeholder="e.g. Dr. Jane Smith")

    if st.button("📄 Generate Report", use_container_width=False):
        report_text = generate_report(detections, duration, patient_name, patient_id, doctor_name)
        report_html = generate_html_preview(detections, duration, patient_name, patient_id, doctor_name)
        
        st.markdown("### 📑 Report Preview")
        st.markdown(report_html, unsafe_allow_html=True)

        # Download button
        buf = io.BytesIO(report_text.encode())
        fname = f"StutterSense_Report_{patient_id or 'patient'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=buf,
            file_name=fname,
            mime="text/plain",
            use_container_width=False,
        )

        # Summary metrics
        st.markdown("---")
        st.markdown("### 📊 Summary Metrics")
        ev_count = len(detections)
        ssi = round(ev_count / max(duration/60, 0.01), 1)
        s1, s2, s3 = st.columns(3)
        for col, val, lbl in zip(
            [s1,s2,s3],
            [ev_count, f"{ssi}/min", f"{duration:.0f}s"],
            ["Total Events","Stutter Rate (SSI)","Audio Duration"],
        ):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="font-size:1.4rem">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        if detections:
            st.pyplot(plot_event_distribution(detections), use_container_width=False)