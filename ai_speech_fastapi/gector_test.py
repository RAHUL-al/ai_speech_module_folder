# import torch
# import torchaudio
# import numpy as np

# model = torch.jit.load("silero_vad/silero-vad/src/silero_vad/data/silero_vad.jit")
# model.eval()

# wav, sr = torchaudio.load("harvard.wav")

# if wav.shape[0] > 1:
#     wav = torch.mean(wav, dim=0, keepdim=True)

# if sr != 16000:
#     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
#     wav = resampler(wav)

# wav = wav / wav.abs().max()
# wav = wav.squeeze()

# SAMPLE_RATE = 16000
# WINDOW_SIZE = 512
# HOP_SIZE = 160
# THRESHOLD = 0.3

# speech_segments = []

# for i in range(0, len(wav) - WINDOW_SIZE + 1, HOP_SIZE):
#     chunk = wav[i:i + WINDOW_SIZE].unsqueeze(0)
    
#     with torch.no_grad():
#         try:
#             prob = model(chunk, SAMPLE_RATE).item()
#         except Exception as e:
#             print(f"Skipping chunk due to error: {e}")
#             continue

#     label = "Speech" if prob > THRESHOLD else "Silence"
#     time = i / SAMPLE_RATE
#     speech_segments.append((time, label, prob))

# print("\nDetected Segments:")
# print([seg for seg in speech_segments if seg[1] == "Speech"])

import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "hafidikhsan/wav2vec2-large-xlsr-53-english-pronunciation-evaluation-aod-real"
)

speech_array, sampling_rate = torchaudio.load("harvard.wav")

if speech_array.shape[0] > 1:
    speech_array = torch.mean(speech_array, dim=0, keepdim=True)

if sampling_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    speech_array = resampler(speech_array)

inputs = feature_extractor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

scores = torch.softmax(logits, dim=-1)
print(f"{scores[0][1].item():.2f}")