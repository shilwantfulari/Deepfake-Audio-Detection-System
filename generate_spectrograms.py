import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

DATASET_PATH = "dataset"
SAVE_PATH = "spectrograms"

def save_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel)

    plt.figure(figsize=(3,3))
    librosa.display.specshow(mel_db, sr=sr)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

for label in ["real","fake"]:
    folder = os.path.join(DATASET_PATH,label)
    save_folder = os.path.join(SAVE_PATH,label)

    os.makedirs(save_folder, exist_ok=True)

    for file in os.listdir(folder):
        if file.lower().endswith((".wav",".mp3",".flac",".ogg")):

            input_path = os.path.join(folder,file)
            print("Processing:", input_path)
            output_path = os.path.join(save_folder,file.replace(".wav",".png"))
            save_spectrogram(input_path,output_path)

print("Spectrogram generation completed.")
