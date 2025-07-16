# 🎤 Emotion Recognition from Speech using MLPClassifier (scikit-learn)

This project detects human emotions from speech using a Multi-Layer Perceptron (MLP) model trained on the RAVDESS dataset. It uses MFCC, chroma, and mel spectrogram features for classification.

---

## 📊 Model Performance

- **Training Accuracy**: 100.00%
- **Test Accuracy**: 86.98%
- **Detected Emotions**: `calm`, `happy`, `fearful`, `disgust`

---

## 📁 Folder Structure

```
Emotion Recognition/
├── emotion_dataset/              # Full RAVDESS dataset (not included in repo due to size)
│   └── emotion-dataset/
├── models/                       # Saved model and pre-processing objects
│   ├── emotion_classification-model.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
├── results/                      # Visualizations and metrics
│   ├── loss_curve.png
│   └── model_metrics.txt
├── sample.wav                    # Sample audio file for prediction
├── Emotion Classification.ipynb # Jupyter notebook (full training workflow)
├── main.py                       # Main script to train and evaluate the model
├── predict.py                    # Script to predict emotion from a .wav file
├── requirements.txt              # Dependencies list
└── README.md                     # Project documentation


```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/nirmal2604/emotion-recognition.git
cd emotion-recognition
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Run Training (Optional — model is already saved)
```bash
python main.py
```

### 4. Predict Emotion from Audio
```bash
python predict.py sample.wav
```

---

## 🗣 Supported Emotions

- calm  
- happy  
- fearful  
- disgust  

> You can extend to others (e.g., angry, sad, neutral) by modifying `observed_emotions` and retraining.

---

## 📈 Sample Output (Loss Curve)

Save a loss curve as `loss_curve.png` and include it in a `graphs/` folder to display like this:

```
Training loss over iterations → 
(Insert loss curve plot here or save using `plt.savefig("graphs/loss_curve.png")`)
```

---

## 📂 Dataset Used

- [RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976)

Place the dataset inside the `emotion_dataset/` directory.

---

## 📜 License

MIT License  
© 2025 Nirmal Modi
