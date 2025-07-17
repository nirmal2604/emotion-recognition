import streamlit as st
import numpy as np
import pickle
import librosa

# Load model and preprocessing tools
model = pickle.load(open("models/emotion_classification-model.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.set_page_config(page_title="Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload a `.wav` audio file to detect the emotion.")

# File uploader
audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

def extract_features(file):
    # Load audio file
    y, sr = librosa.load(file, sr=None)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    try:
        # Extract features
        features = extract_features(audio_file)
        features_scaled = scaler.transform(features)
        # Predict emotion
        pred = model.predict(features_scaled)
        emotion = label_encoder.inverse_transform(pred)[0]
        st.success(f"Predicted Emotion: **{emotion}**")
    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")
