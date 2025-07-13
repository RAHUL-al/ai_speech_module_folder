# import torch
# import torchaudio
# import numpy as np

# model = torch.jit.load("silero_vad.jit")
# model.eval()

# wav, sr = torchaudio.load("harvard.wav")
# if sr != 16000:
#     raise ValueError("Sample rate must be 16kHz")

# wav = wav.squeeze()

# speech_probs = model(wav, sr)

# speech_threshold = 0.5
# speech = speech_probs.numpy() > speech_threshold

# for i, is_speech in enumerate(speech):
#     print(f"{i * 0.02:0.2f}s - {'Speech' if is_speech else 'Silence'}")


# vad_detector.py

import torch
import torchaudio
import numpy as np
import os

# Load model once globally
model_path = os.path.join(os.path.dirname(__file__), "silero_vad.jit")
model = torch.jit.load(model_path)
model.eval()

def detect_speech_regions(audio_path, threshold=0.5):
    wav, sr = torchaudio.load(audio_path)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)

    wav = wav.squeeze()
    speech_probs = model(wav, 16000)
    speech = speech_probs.numpy() > threshold

    results = []
    for i, is_speech in enumerate(speech):
        timestamp = round(i * 0.02, 2)
        results.append({
            "time": timestamp,
            "label": "Speech" if is_speech else "Silence"
        })

    return results
