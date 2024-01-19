import os
import threading
from flask import Flask, render_template, request, jsonify
import pyaudio
import wave
import whisper
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Параметры аудиозаписи
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORDING = False

last_index = 0
frames = []
recording_thread = None


@app.route('/')
def index():
    return render_template('old/index.html')


@app.route('/start_recording', methods=['POST'])
def start_recording():
    global RECORDING, frames, recording_thread
    if RECORDING:
        return jsonify({"message": "Already recording"})

    RECORDING = True
    frames = []

    def record():
        local_audio = pyaudio.PyAudio()
        stream = local_audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        while RECORDING:
            frames.append(stream.read(CHUNK, exception_on_overflow=False))
        stream.stop_stream()
        stream.close()
        local_audio.terminate()

    recording_thread = threading.Thread(target=record)
    recording_thread.start()

    return jsonify({"message": "Recording started"})


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global RECORDING, frames, last_index, recording_thread
    RECORDING = False
    if recording_thread:
        recording_thread.join()

    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    last_index += 1
    filename = f"uploads/recorded_{last_index}.wav"

    local_audio = pyaudio.PyAudio()
    with wave.open(filename, 'wb') as waveFile:
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(local_audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
    local_audio.terminate()

    return jsonify({"message": f"Recording stopped and saved as {filename}"})


def transcribe_audio(audio_file_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file_path)
        return result['text']
    except Exception as e:
        return f"An error occurred: {str(e)}"


@app.route('/transcription')
def transcription_page():
    files = os.listdir('uploads')
    return render_template('old/transcription.html', files=files)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.form.get('audio_file')
    audio_file_path = os.path.join('uploads', audio_file)
    transcription = transcribe_audio(audio_file_path)
    files = os.listdir('uploads')
    return render_template('old/transcription.html', transcription=transcription, files=files)


@app.route('/emotion')
def emotion_page():
    files = os.listdir('uploads')
    return render_template('old/emotion.html', files=files)


def analyze_emotions(text):
    try:
        tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        predicted_sentiment = predictions.item()

        # Здесь вы можете изменить логику определения эмоций
        if predicted_sentiment == 0:
            emotion_label = "Очень негативная"
        elif predicted_sentiment == 1:
            emotion_label = "Негативная"
        elif predicted_sentiment == 2:
            emotion_label = "Нейтральная"
        elif predicted_sentiment == 3:
            emotion_label = "Позитивная"
        else:
            emotion_label = "Очень позитивная"

        return emotion_label
    except Exception as e:
        return f"An error occurred: {str(e)}"


@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    audio_file = request.form.get('audio_file')
    audio_file_path = os.path.join('uploads', audio_file)
    transcription = transcribe_audio(audio_file_path)
    emotion = analyze_emotions(transcription)  # Вызов функции анализа эмоций
    files = os.listdir('uploads')
    return render_template('old/emotion.html', emotion=emotion, files=files)


if __name__ == '__main__':
    app.run(debug=True)
