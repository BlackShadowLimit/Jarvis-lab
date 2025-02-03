
from TTS.api import TTS
import sounddevice as sd
import numpy as np

device = "cpu"
print(f"Using device: {device}")


model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False).to(device)
Jarvis_voice = "p236"

def speak(text):
    wav = model.tts(text=text, speaker=Jarvis_voice)
    wav_np = np.array(wav)
    sd.play(wav_np, samplerate=22050)
    sd.wait()



