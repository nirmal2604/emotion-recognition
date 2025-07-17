import streamlit as st
import numpy as np
import pickle
import librosa
import soundfile # Import soundfile for the extract_feature function

# --- Feature Extraction Function (Copy from your training script - MUST be identical) ---
# This function must be the exact same as the one used in your training script (main.py)
# to ensure the same number and type of features are extracted.
def extract_feature(file_name, mfcc=True, chroma=True, mel=True, augment=False):
    # soundfile.SoundFile expects a file path or a file-like object that it can read
    # For Streamlit's UploadedFile, you need to save it temporarily or use its read() method
    # and then pass it to soundfile.SoundFile.
    # A simpler approach for Streamlit is to use librosa.load directly with the BytesIO object.

    # Streamlit's file_uploader returns a BytesIO object, not a file path.
    # librosa.load can handle BytesIO objects directly.
    X, sample_rate = librosa.load(file_name, sr=None)

    # Augmentation should NOT be applied during prediction
    if augment: # This block should always be False for prediction
        X = librosa.effects.pitch_shift(X, sr=sample_rate, n_steps=np.random.uniform(-2, 2))
        X = librosa.effects.time_stretch(X, rate=np.random.uniform(0.8, 1.2))

    # Compute STFT only if chroma features are requested
    stft = np.abs(librosa.stft(X)) if chroma else None
    result = np.array([])

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_features))
    if mel:
        mel_features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_features))
    return result

# --- Load model and preprocessing tools ---
# Ensure correct paths to your models/ directory
try:
    model = pickle.load(open("models/emotion_classification-model.pkl", "rb"))
    label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    st.sidebar.success("Model components loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading model components: {e}. Make sure .pkl files are in the 'models/' directory.")
    st.stop() # Stop the app if model loading fails

st.set_page_config(page_title="Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload a `.wav` audio file to detect the emotion.")

# File uploader
audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    st.write("Processing audio...")
    try:
        # Extract features using the full, correct feature extraction function
        # Pass mfcc=True, chroma=True, mel=True to match training
        # Pass augment=False for prediction
        features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True, augment=False)

        # DEBUGGING: Print the shape of extracted features to Streamlit logs
        st.write(f"Debug: Shape of extracted features: {features.shape}")

        # Reshape the features for the scaler (from 1D to 2D: (1, 180))
        features_reshaped = features.reshape(1, -1)

        # DEBUGGING: Print the shape after reshaping
        st.write(f"Debug: Shape of features after reshape: {features_reshaped.shape}")

        # Scale the features using the loaded scaler
        features_scaled = scaler.transform(features_reshaped)

        # Predict emotion
        pred = model.predict(features_scaled)
        emotion = label_encoder.inverse_transform(pred)[0]

        st.success(f"Predicted Emotion: **{emotion}**")

    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")
        st.error("Please check the audio file format and ensure it's not corrupted.")

