import os
import glob
import numpy as np
import soundfile
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle

# Emotion labels
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Feature extraction
def extract_feature(file_name, mfcc=True, chroma=True, mel=True, augment=False):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        if augment:
            X = librosa.effects.pitch_shift(X, sr=sample_rate, n_steps=np.random.uniform(-2, 2))
            X = librosa.effects.time_stretch(X, rate=np.random.uniform(0.8, 1.2))

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

# Data loader
def load_data(test_size=0.25):
    x, y = [], []

    for file in glob.glob("emotion_dataset/emotion-dataset/Actor_*/*"):
        file_name = os.path.basename(file)
        emotion = emotions.get(file_name.split("-")[2])
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True, augment=False)
        x.append(feature)
        y.append(emotion)

    x_train, x_test, y_train, y_test = train_test_split(np.array(x), y, test_size=test_size, random_state=9)

    for file in glob.glob("emotion_dataset/emotion-dataset/Actor_*/*"):
        file_name = os.path.basename(file)
        emotion = emotions.get(file_name.split("-")[2])
        if emotion not in observed_emotions:
            continue
        for _ in range(1):
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True, augment=True)
            x_train = np.vstack([x_train, feature])
            y_train.append(emotion)

    return x_train, x_test, y_train, y_test

# Load and preprocess
x_train, x_test, y_train, y_test = load_data()
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model initialization
model = MLPClassifier(
    hidden_layer_sizes=(1024, 512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=0.00005,
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=1200,
    random_state=42,
    verbose=True
)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, x_train, y_train, cv=skf, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(scores) * 100:.2f}% ± {np.std(scores) * 100:.2f}%")

print(f"Training samples: {x_train.shape[0]}, Testing samples: {x_test.shape[0]}")
print(f"Features extracted: {x_train.shape[1]}")

# Train and evaluate
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
train_acc = model.score(x_train, y_train)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.savefig("confusion_matrix.png")
plt.close()

# Loss curve
plt.plot(model.loss_curve_)
plt.title("MLP Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve.png")
plt.close()

# Save metrics
with open("model_metrics.txt", "w") as f:
    f.write(f"Training Accuracy: {train_acc * 100:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
with open("emotion_classification-model.pkl", "wb") as f:
    pickle.dump(model, f)
