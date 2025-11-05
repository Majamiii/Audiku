# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import csv
import numpy as np
import librosa
import whisper
import ollama
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)
CORS(app)  # This allows your React app to call this API

# Create temp directory for uploaded files
os.makedirs('./temp', exist_ok=True)

class GetText:
    @staticmethod
    def description(audio_path: str) -> str:
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

        result = whisper_model.transcribe(audio_path, language="en")
        speech_text = result["text"]

        chunk_duration = 3
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
        f"You are a poetic assistant. Respond with a sentimental haiku (5-7-5 syllable structure) based on this description: {description} \
        Output ONLY THE HAIKU. Always generate sentimental haikus that capture the mood and emotion of the input. \
        Output the haiku and nothing else."
    )

    response = ollama.chat(
        model="haiku",
        messages=[{"role": "user", "content": prompt}]
    )

    haiku = response.message["content"].replace('\\n', '\n')
    return haiku

@app.route('/api/generate-haiku', methods=['POST'])
def generate_haiku():
    try:
        # Check if file was uploaded
        if 'audio' in request.files:
            audio = request.files['audio']
            audio_path = f"./temp/{audio.filename}"
            audio.save(audio_path)
        # Check if path was provided
        elif request.json and 'path' in request.json:
            audio_path = request.json['path']
        else:
            return jsonify({'error': 'No audio file or path provided'}), 400
        
        # Generate description and haiku
        description = GetText.description(audio_path)
        haiku = generate_haiku_from_audio(audio_path)
        
        return jsonify({
            'description': description,
            'haiku': haiku
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)