# pip install openai-whisper soundfile

import whisper

model = whisper.load_model("base")  # faster models: tiny, base
result = model.transcribe("./audio_samples/rock.mp3", language="en")
print(result["text"])  # this text will feed into your haiku generator