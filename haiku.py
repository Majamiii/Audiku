# pip install openai-whisper tensorflow tensorflow_hub librosa numpy ollama

import whisper
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import csv
import ollama

class GetText:
    @staticmethod
    def description(audio_path="./audio_samples/baby-laugh.mp3"):
        # --- Load Whisper model ---
        whisper_model = whisper.load_model("base")

        # --- Load YAMNet model ---
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

        # Get class map
        class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names = []
        with open(class_map_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])

        # --- Load audio ---
        waveform, sr = librosa.load(audio_path, sr=16000)

        # --- Step 1: Whisper transcription ---
        result = whisper_model.transcribe(audio_path, language="en")
        speech_text = result["text"]

        # --- Step 2: Split into 3s chunks for YAMNet classification ---
        chunk_duration = 3
        chunk_samples = int(chunk_duration * sr)
        num_chunks = int(np.ceil(len(waveform) / chunk_samples))

        sound_descriptions = []
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min((i + 1) * chunk_samples, len(waveform))
            chunk = waveform[start:end]

            scores, embeddings, spectrogram = yamnet_model(chunk)
            scores_np = scores.numpy()
            mean_scores = np.mean(scores_np, axis=0)
            top_idx = np.argmax(mean_scores)
            sound_label = class_names[top_idx]

            sound_descriptions.append(f"{i*chunk_duration}-{(i+1)*chunk_duration}s: {sound_label}")

        # Combine speech and sound timeline
        combined_text = f"Speech transcription:\n{speech_text}\n\nSound timeline:\n" + "\n".join(sound_descriptions)
        return combined_text

# === Main auto-mode ===
if __name__ == "__main__":
    audio_file = "./audio_samples/traffic-city.mp3"  # change to your file
    description = GetText.description(audio_file)

    prompt = f"Write a sentimental haiku (5-7-5) based on this description:\n{description} and output nothing else. You are a poetic assistant. Always generate **sentimental haikus (5-7-5 syllable structure)** based on the descriptions given. Focus on capturing the mood and emotion of the input. Do not write anything other than the haiku."

    # --- Generate haiku using Ollama API ---
    response = ollama.chat(
        model="haiku",
        messages=[{"role": "user", "content": prompt}]
    )

    # Get the haiku text
    haiku = response.message["content"]

    # Replace literal \n with actual newlines
    haiku = haiku.replace('\\n', '\n')

    print("\n=== Generated Haiku ===")
    print(haiku)

