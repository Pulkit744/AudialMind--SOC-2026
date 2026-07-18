# VoxSense – Emotion-Aware Speech Interaction Assistant

VoxSense is an end-to-end speech interaction system developed as the AudialMind Seasons of Code 2026 capstone project. The system takes a speech audio file as input and performs:

- Automatic Speech Recognition (ASR)
- Speech Emotion Recognition (SER)
- Returns both results in a single JSON output.

---

## Project Structure

```
final_project/
│
├── voxsense.py
├── notebook.ipynb
├── ser_model.pkl
├── ser_scaler.pkl
├── README.md
├── report.pdf
├── charts/
├── archive/          # RAVDESS dataset
└── LibriSpeech/      # ASR evaluation dataset
```

---

## Features

- Speech Emotion Recognition using eGeMAPS features and an SVM (RBF kernel)
- Speech-to-Text using Facebook Wav2Vec2 Base 960h
- Combined JSON output
- Cross-validation evaluation using UAR
- ASR evaluation using WER and CER

---

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas matplotlib librosa librosa-display
pip install scikit-learn
pip install opensmile
pip install torch transformers
pip install jiwer
pip install joblib
```

---

## Datasets

### Speech Emotion Recognition

- RAVDESS

### Speech Recognition Evaluation

- LibriSpeech test-clean

---

## Running VoxSense

Run the script using:

```bash
python voxsense.py --file path_to_audio.wav
```

Example:

```bash
python voxsense.py --file archive/Actor_01/03-01-01-01-01-01-01.wav
```

---

## Example Output

```json
{
    "transcript": "KIDS ARE TALKING BY THE DOOR",
    "emotion": {
        "label": "neutral",
        "confidence": 0.3674
    },
    "asr_model": "facebook/wav2vec2-base-960h",
    "ser_model": "eGeMAPS + SVM",
    "processing_time_sec": 3.85
}
```

---

## Models Used

### Speech Emotion Recognition

- OpenSMILE eGeMAPSv02
- StandardScaler
- SVM (RBF Kernel)

### Automatic Speech Recognition

- facebook/wav2vec2-base-960h

---

## Evaluation Metrics

Speech Emotion Recognition:

- Unweighted Average Recall (UAR)

Automatic Speech Recognition:

- Word Error Rate (WER)
- Character Error Rate (CER)

---

## Author

Pulkit Gupta

AudialMind – Seasons of Code 2026
