from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch
import torchaudio

# Load feature extractor and model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2ForSequenceClassification.from_pretrained("hafidikhsan/wav2vec2-large-xlsr-53-english-pronunciation-evaluation-aod-real")

# Load audio file
speech_array, sampling_rate = torchaudio.load("final_output.wav")
if sampling_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    speech_array = resampler(speech_array)

# Preprocess
inputs = feature_extractor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt")

# Predict
with torch.no_grad():
    logits = model(**inputs).logits

# Get score
scores = torch.softmax(logits, dim=-1)
print("🗣 Pronunciation score:", scores)
