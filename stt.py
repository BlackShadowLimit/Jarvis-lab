import pyaudio
import wave
import numpy as np
import whisper
from whisper.transcribe import transcribe

def is_silent(data, threshold=500):
    audio_data = np.frombuffer(data, dtype=np.int16)
    volume = np.abs(audio_data).mean()
    return volume < threshold

def record_until_silence(silence_threshold, output_file):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 16000
    silent_duration = 0
    frames = []
    pole = True
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    print("start_record")

    try:
        while True:
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

            if is_silent(data):
                silent_duration += chunk / rate
            else:
                silent_duration = 0
                pole = False

            if silent_duration >= silence_threshold:
                break
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    if pole:
        return None

    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

    return output_file

def transcribe_audio(file_path):
    base_model = whisper.load_model('tiny', device='cpu')
    transcription = base_model.transcribe(file_path)
    if transcription and transcription['text']:
        return str(transcription['text'])
    else:
        return None

def main():
    input_text = ""
    silence_threshold = 1
    output_file = "user-input.wav"
    audio_file = record_until_silence(silence_threshold, output_file)
    if audio_file:
        result_text = transcribe_audio(audio_file)
        input_text += str(result_text)
    return input_text

if __name__ == "__main__":
    print(main())

