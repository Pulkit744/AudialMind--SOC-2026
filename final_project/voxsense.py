import argparse
import json
import time
import joblib
import librosa
import numpy as np
import opensmile
import torch
import os

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)

print("Loading models...")

ser_model = joblib.load("ser_model.pkl")
ser_scaler = joblib.load("ser_scaler.pkl")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

asr_model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

print("Models loaded.")

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

def transcribe_audio(audio_path):

    speech, sr = librosa.load(
        audio_path,
        sr=16000
    )

    inputs = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = asr_model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    transcript = processor.batch_decode(
        predicted_ids
    )[0]

    return transcript

def predict_emotion(audio_path):

    features = smile.process_file(audio_path)

    X = features.to_numpy()

    X = ser_scaler.transform(X)

    prediction = ser_model.predict(X)[0]

    confidence = float(
        ser_model.predict_proba(X)[0].max()
    )

    return prediction, confidence

def main(audio_path):

    start = time.time()

    emotion, confidence = predict_emotion(audio_path)

    transcript = transcribe_audio(audio_path)

    result = {
        "transcript": transcript,
        "emotion": {
            "label": emotion,
            "confidence": round(confidence, 4)
        },
        "asr_model": "facebook/wav2vec2-base-960h",
        "ser_model": "eGeMAPS + SVM",
        "processing_time_sec": round(time.time() - start, 2)
    }

    print(json.dumps(result, indent=4))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="VoxSense: Speech Emotion Recognition and ASR"
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to input audio file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.file):
    print(f"Error: File '{args.file}' not found.")
    exit(1)
    
    main(args.file)