import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import sounddevice as sd
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
from attention_layer import Attention

# ================= LOAD MODEL =================
model = load_model("deepfake_model.keras",
                   custom_objects={'Attention':Attention})

IMG_SIZE = 128

st.title(" AI Fake Voice Detector ")

# ================= AUDIO PROCESSING =================
def audio_to_spectrogram(audio_path):

    # Load audio safely
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # 🔥 Remove silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # 🔥 Normalize
    y = librosa.util.normalize(y)

    # 🔥 Pre-emphasis (fixes WhatsApp compression)
    y = librosa.effects.preemphasis(y)

    # Detect compression (simple calibration trick)
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel)

    plt.figure(figsize=(3,3))
    librosa.display.specshow(mel_db, sr=sr)
    plt.axis('off')
    plt.savefig("temp.png",bbox_inches='tight',pad_inches=0)
    plt.close()

    img=cv2.imread("temp.png")
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img=img/255.0
    img=np.expand_dims(img,axis=0)

    return img, spectral_flatness, y

# ================= UPLOAD =================
audio_file = st.file_uploader(
    "Upload Audio (wav / mp3 / ogg / flac / m4a)",
    type=["wav","mp3","ogg","flac","m4a"]
)

# ================= RECORD =================
if st.button("🎤 Record 5 sec Voice"):
    fs=16000
    seconds=5
    st.write("Recording...")
    recording=sd.rec(int(seconds*fs),samplerate=fs,channels=1)
    sd.wait()
    write("recorded.wav",fs,recording)
    st.success("Recording Done")
    audio_file="recorded.wav"

# ================= PREDICTION =================
if audio_file:

    if isinstance(audio_file,str):
        path=audio_file
    else:
        path="uploaded_audio"
        with open(path,"wb") as f:
            f.write(audio_file.read())

    features, flatness, waveform = audio_to_spectrogram(path)

    pred=model.predict(features)[0][0]

    # 🔥 CALIBRATION LOGIC
    prob=float(0.7*pred + 0.3*0.5)

    # If audio looks compressed (like WhatsApp), reduce fake bias slightly
    if flatness > 0.2:
        prob = prob * 0.85

    st.write(f"### Fake Probability: {prob*100:.2f}%")

    # ================= PROFESSIONAL VISUALS =================
    st.subheader("Confidence Meter")
    st.progress(min(int(prob*100),100))

    # Waveform display
    st.subheader("Audio Waveform")
    fig, ax = plt.subplots()
    ax.plot(waveform)
    st.pyplot(fig)

    # ================= DECISION SYSTEM =================
    if prob > 0.70:
        st.error("⚠️ AI GENERATED VOICE DETECTED")

    elif prob > 0.60:
        st.warning("🤔 UNCERTAIN RESULT (Compressed or Noisy Audio)")

    else:
        st.success("✅ REAL HUMAN VOICE")

    st.caption("Higher fake probability means voice patterns resemble AI-generated audio.")
