#  Deepfake Audio Detection System

An AI-based deepfake voice detection system that analyzes audio recordings and predicts whether the voice is real or AI-generated.

## Features
- Real vs Fake voice detection
- CNN + LSTM + Attention deep learning model
- Mel Spectrogram feature extraction
- Real-time Streamlit web application
- Fake probability score with confidence visualization

##  Tech Stack
Python, TensorFlow, Librosa, OpenCV, Streamlit

## How to Run
1. Train model:
python train_model.py

2. Run web app:
streamlit run realtime_app.py

## Project Structure
- train_model.py – model training
- realtime_app.py – web application
- generate_spectrograms.py – preprocessing
- attention_layer.py – attention layer

Dataset: Kaggle Deepfake Audio Dataset
