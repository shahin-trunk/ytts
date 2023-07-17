import os

wav_file = "/data/asr/workspace/audio/tts/models/speaker_encoding/musan/speech/us-gov/speech-us-gov-0135.wav"
additive_path = "/data/asr/workspace/audio/tts/models/speaker_encoding/musan/"
noise_dir = wav_file.replace(additive_path, "").split(os.sep)[0]
print(noise_dir)
