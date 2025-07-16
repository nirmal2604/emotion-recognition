import sys
import numpy as np
import soundfile
import librosa
import pickle

# -----------------------------
# Validate input
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python predict.py <audio_file.wav>")
    sys.exit(1)

file_path = sys.argv[1]

# -----------------------------
# Load saved model components
# -----------------------------
with open("models/emotion_classification-model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -----------------------------
# Feature extraction
# -----------------------------
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        stft = np.abs(librosa.stft(X)) if chroma else None
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# -----------------------------
# Predict emotion
# -----------------------------
features = extract_feature(file_path).reshape(1, -1)
features = scaler.transform(features)
prediction = model.predict(features)
emotion = le.inverse_transform(prediction)

print(f"Predicted Emotion: {emotion[0]}")
