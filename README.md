# STUTTERSENSE-CLINICAL-AI-SPEECH-FLUENCY-ANALYZER

## Overview

StutterSense is an AI-powered clinical speech analysis platform designed to automate the detection and classification of stuttering dysfluencies from speech recordings. The system combines audio signal processing, deep learning, and interactive visualization to assist Speech-Language Pathologists (SLPs) in conducting objective and reproducible speech assessments.

The platform analyzes speech recordings, identifies dysfluency events, computes severity metrics, and generates structured clinical reports to support diagnostic workflows.

---

## Key Features

✅ Multi-class Stuttering Classification

* No Stutter
* Prolongation
* Block
* Sound Repetition
* Word Repetition

✅ Audio Analysis

* Audio file upload support
* Live microphone recording support
* MFCC feature extraction using Librosa

✅ Deep Learning-Based Detection

* Hybrid CNN-BiLSTM architecture
* Sliding-window inference for event detection
* Confidence-based predictions

✅ Clinical Insights

* Stutter Severity Index (SSI) calculation
* Timestamped dysfluency event tracking
* Dominant stutter type identification
* Clinical recommendation generation

✅ Interactive Dashboard

* Real-time spectrogram visualization
* Event distribution charts
* Model training metrics visualization
* Confusion matrix and classification reports

✅ Report Generation

* Patient information integration
* Diagnostic summary creation
* Downloadable clinical reports

---

## System Architecture

### Audio Processing

1. Speech audio is loaded and preprocessed.
2. MFCC features are extracted using Librosa.
3. Features are normalized using StandardScaler.

### Deep Learning Pipeline

Input Audio

→ MFCC Feature Extraction

→ CNN Layers (Local Feature Learning)

→ Bidirectional LSTM Layers (Temporal Pattern Learning)

→ Softmax Classification

→ Dysfluency Detection & Clinical Analysis

---

## Model Architecture

* Conv1D (64 Filters)
* Batch Normalization
* MaxPooling1D
* Dropout (0.3)
* Bidirectional LSTM (64 Units)
* Batch Normalization
* Dropout
* Bidirectional LSTM (32 Units)
* Batch Normalization
* Dropout
* Dense (64, ReLU)
* Dense (5, Softmax)

---

## Technologies Used

### Programming Language

* Python

### Machine Learning & Deep Learning

* TensorFlow
* Keras
* Scikit-learn
* Imbalanced-learn (SMOTE)

### Audio Processing

* Librosa

### Data Visualization

* Matplotlib

### Web Application

* Streamlit

### Datasets

* SEP-28k Dataset
* UCLASS Dataset

---

## Project Structure

```text
StutterSense/
│
├── app.py
├── model/
├── training/
├── preprocessing/
├── reports/
├── datasets/
├── assets/
├── requirements.txt
└── README.md
```

---

## Dataset Information

This project utilizes:

* SEP-28k (Stuttering Events in Podcasts)
* UCLASS (University of Central Lancashire Archive of Stuttered Speech)

### Important Note

Audio recordings and large CSV dataset files are **not included in this repository** because GitHub imposes a **100 MB file size limit**.

To run this project locally:

1. Download the required datasets separately from their official sources.
2. Place the audio files and CSV datasets inside the appropriate dataset directories.
3. Update dataset paths in the configuration files if required.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/StutterSense.git

cd StutterSense
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Future Enhancements

* Real-time streaming speech analysis
* Transformer-based speech classification models
* Enhanced clinical reporting formats
* Cloud deployment support
* Multi-language stuttering assessment
* Integration with Electronic Health Record (EHR) systems

---

## Clinical Applications

StutterSense is designed to support Speech-Language Pathologists by providing:

* Objective dysfluency assessment
* Automated speech analysis
* Severity estimation
* Clinical report generation
* Evidence-based diagnostic support

---
