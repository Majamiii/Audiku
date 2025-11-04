# pip install openai-whisper tensorflow tensorflow_hub librosa numpy ollama

import csv
import numpy as np
import librosa
import whisper
import ollama
import tensorflow as tf
import tensorflow_hub as hub

class GetText:
    @staticmethod
    def description(audio_path: str = "./audio_samples/baby-laugh.mp3") -> str:
        """Return a combined speech + sound description from audio."""
        whisper_model = whisper.load_model("base")
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

        # get all the labels
        class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names = []
        with open(class_map_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])

        waveform, sr = librosa.load(audio_path, sr=16000)

        result = whisper_model.transcribe(audio_path, language="en")        # speech
        speech_text = result["text"]

        chunk_duration = 3                              # sound  - split into 3a chunks
        chunk_samples = int(chunk_duration * sr)
        num_chunks = int(np.ceil(len(waveform) / chunk_samples))

        sound_descriptions = []
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min((i + 1) * chunk_samples, len(waveform))
            chunk = waveform[start:end]

            scores, embeddings, spectrogram = yamnet_model(chunk)
            mean_scores = np.mean(scores.numpy(), axis=0)
            top_idx = np.argmax(mean_scores)
            sound_label = class_names[top_idx]

            sound_descriptions.append(f"{i*chunk_duration}-{(i+1)*chunk_duration}s: {sound_label}")

        combined_text = f"Speech transcription:\n{speech_text}\n\nSound timeline:\n" + "\n".join(sound_descriptions)
        return combined_text

def generate_haiku_from_audio(audio_file: str) -> str:
    description = GetText.description(audio_file)

    prompt = (
        f"Write a sentimental haiku (5-7-5 syllable structure) based on this description: {description} \
        Output ONLY THE HAIKU. You are a poetic assistant. Always generate sentimental haikus that capture the mood and emotion of the input. \
        Do not include ANYTHING OTHER than the haiku. \
        A haiku has STRICTLY 3 lines with a 5-7-5 syllable count: Line 1: 5 syllables, Line 2: 7 syllables, Line 3: 5 syllables"
    )

    response = ollama.chat(
        model="haiku",
        messages=[{"role": "user", "content": prompt}]
    )

    haiku = response.message["content"].replace('\\n', '\n')
    return haiku


# example usage
if __name__ == "__main__":
    audio_file = "./audio_samples/rock.mp3"
    haiku = generate_haiku_from_audio(audio_file)
    print(haiku)

