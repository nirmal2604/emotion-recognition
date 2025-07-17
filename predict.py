import os
import sys # Import sys for command-line arguments
import numpy as np
import soundfile
import librosa
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

# --- Feature Extraction Function (Must be IDENTICAL to training) ---
# Ensure this function is identical to the one used during training,
# including the 'augment' parameter, even if it's set to False for prediction.
def extract_feature(file_name, mfcc=True, chroma=True, mel=True, augment=False):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        # Augmentation should typically NOT be applied during prediction
        # as it would change the input characteristics and lead to mismatch
        if augment:
            X = librosa.effects.pitch_shift(X, sr=sample_rate, n_steps=np.random.uniform(-2, 2))
            X = librosa.effects.time_stretch(X, rate=np.random.uniform(0.8, 1.2))

        # Compute STFT only if chroma features are requested
        # Using np.abs(librosa.stft(X)) as it was in your training script
        stft = np.abs(librosa.stft(X)) if chroma else None
        result = np.array([])

        if mfcc:
            # MFCC features: n_mfcc=40, as per your training script
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            # Chroma features: default n_chroma=12
            chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_features))
        if mel:
            # Mel-spectrogram features: default n_mels=128
            mel_features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_features))
    return result

# --- Load the trained model, scaler, and label encoder ---
try:
    # Ensure paths are correct based on your repo structure (models/ directory)
    with open("models/emotion_classification-model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    print("Model, scaler, and label encoder loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Make sure 'emotion_classification-model.pkl', 'scaler.pkl', and 'label_encoder.pkl' are in the 'models/' directory.")
    sys.exit(1) # Exit if essential files are missing

# -----------------------------
# Validate input (from your original predict.py)
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <audio_file.wav>")
        sys.exit(1)

    file_path = sys.argv[1]

    # -----------------------------
    # Predict emotion
    # -----------------------------
    try:
        # 1. Extract features using the SAME parameters as training
        # mfcc=True, chroma=True, mel=True must be set to match the training
        # augment=False is crucial for prediction
        features = extract_feature(file_path, mfcc=True, chroma=True, mel=True, augment=False)

        # DEBUGGING: Print the shape of extracted features
        print(f"Shape of extracted features: {features.shape}")

        # 2. Reshape the features for the scaler
        # StandardScaler expects a 2D array (n_samples, n_features)
        # If 'features' is a 1D array (180,), reshape it to (1, 180)
        features = features.reshape(1, -1) # Reshapes to (1, 180)

        # DEBUGGING: Print the shape after reshaping
        print(f"Shape of features after reshape: {features.shape}")

        # 3. Scale the features using the loaded scaler
        scaled_features = scaler.transform(features)

        # 4. Make prediction
        prediction = model.predict(scaled_features)
        emotion = le.inverse_transform(prediction)

        print(f"Predicted Emotion: {emotion[0]}")

    except soundfile.LibsndfileError as e:
        print(f"Error processing audio file: {e}. Ensure it's a valid WAV file.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        sys.exit(1)

