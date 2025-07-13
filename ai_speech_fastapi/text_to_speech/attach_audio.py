from pydub import AudioSegment
import os

# Path where your WAV files are saved
folder_path = "./"  # or specify your folder

# Get all WAV files sorted by name (optional)
wav_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.wav')])
print(wav_files)
# Start with an empty audio segment
combined = AudioSegment.empty()

# Combine all audio files
for file in wav_files:
    audio = AudioSegment.from_wav(os.path.join(folder_path, file))
    combined += audio

# Export final combined WAV
combined.export("final_output.wav", format="wav")
print("✅ Combined audio saved as 'final_output.wav'")
