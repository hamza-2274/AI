import streamlit as st
import librosa
import numpy as np
import torch
import torch.nn as nn

# ================================
# 1. Define class labels and feature extraction
# ================================
CLASS_NAMES = ["Fan", "Pump", "Gearbox"]
MODEL_PATH = "machine_classifier.pt"

def extract_features(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

    return np.hstack((mfcc_scaled, spectral_centroid, spectral_bandwidth, zero_crossing_rate, rms, chroma_stft))

# ================================
# 2. Define Model Class (Must match training code!)
# ================================
class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ================================
# 3. Streamlit App Layout
# ================================
st.set_page_config(page_title="Machine Sound Classifier", page_icon="🔊", layout="centered")

st.title("🔊 Machine Sound Classifier")
st.markdown("Upload a `.wav` file of a machine sound and I’ll tell you if it’s a **Fan**, **Pump**, or **Gearbox**.")

uploaded_file = st.file_uploader("📁 Upload your .wav file here", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    
    try:
        # Extract features from uploaded file
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        features = extract_features("temp.wav")
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Load model
        model = MLPClassifier(input_size=input_tensor.shape[1], num_classes=3)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            predicted_label = CLASS_NAMES[prediction]

        st.success(f"🎯 Prediction: **{predicted_label}**")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("👆 Please upload a `.wav` file to get started.")
