# pip install openai-whisper tensorflow tensorflow_hub librosa numpy

import whisper
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import csv

# --- Load Whisper model ---
whisper_model = whisper.load_model("base")

# --- Load YAMNet model ---
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Get path to built‑in class map
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = []
with open(class_map_path, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'])

# --- Load audio ---
audio_path = "./audio_samples/baby-laugh.mp3"
waveform, sr = librosa.load(audio_path, sr=16000)  # YAMNet expects 16kHz

# --- Step 1: Transcribe speech with Whisper ---
result = whisper_model.transcribe(audio_path, language="en")
speech_text = result["text"]

# --- Step 2: Split audio into chunks for classification ---
chunk_duration = 3  # seconds
chunk_samples = int(chunk_duration * sr)
num_chunks = int(np.ceil(len(waveform) / chunk_samples))

sound_descriptions = []
for i in range(num_chunks):
    start = i * chunk_samples
    end = min((i + 1) * chunk_samples, len(waveform))
    chunk = waveform[start:end]

    # Expand dims if needed (model may want shape [samples])
    scores, embeddings, spectrogram = yamnet_model(chunk)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)
    top_idx = np.argmax(mean_scores)
    sound_label = class_names[top_idx]

    sound_descriptions.append(f"{i*chunk_duration}-{(i+1)*chunk_duration}s: {sound_label}")

# --- Step 3: Print outputs ---
print("=== Speech Transcription ===")
print(speech_text)
print("\n=== Sound Descriptions ===")
for desc in sound_descriptions:
    print(desc)


# for llama model

timeline_text = []

for i, desc in enumerate(sound_descriptions):
    # If your Whisper transcription has timestamps, you could also align segments
    # For simplicity, we'll just append the sound description
    timeline_text.append(desc)

# Add speech at the top or as first element
combined_text = f"Speech transcription:\n{speech_text}\n\nSound timeline:\n" + "\n".join(timeline_text)