import json
import logging
import os
import random
import uuid

import torchaudio
from google.cloud import texttospeech

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("GG")

SPEAKERS = ["ar-XA-Wavenet-C", "ar-XA-Wavenet-B", "ar-XA-Wavenet-D", "ar-XA-Wavenet-A"]
ENV = ["headphone-class-device", "small-bluetooth-speaker-class-device"]

RUN_INDEX = 1
SAMPLE_RATE = 44100
GEN_ALL_TEXT = True

manifest_dir = "data/tts/manifest/google"
wav_dir = "data/tts/wav/google"
cred_dir = "data/tts/cred/google"
text_dir = "data/tts/text/google"

os.makedirs(manifest_dir, exist_ok=True)
os.makedirs(wav_dir, exist_ok=True)
os.makedirs(cred_dir, exist_ok=True)
os.makedirs(text_dir, exist_ok=True)

manifest_path = os.path.join(manifest_dir, f"manifest_{RUN_INDEX}.json")
text_path = os.path.join(text_dir, f"src_text_{RUN_INDEX}.txt")

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(cred_dir, "application_default_credentials.json")

# Instantiates a client
client = texttospeech.TextToSpeechClient()


def get_synthesis_meta(_text: str, _spk_index, _env_index):
    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code="ar-XA",
        name=SPEAKERS[_spk_index],
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        effects_profile_id=[ENV[_env_index]]
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    u_fid = str(uuid.uuid4())
    target_wav_dir = os.path.join(wav_dir, f"spk_{_spk_index}/env_{_env_index}")
    os.makedirs(target_wav_dir, exist_ok=True)
    target_wav_path = os.path.join(target_wav_dir, f"{u_fid}_16b_{SAMPLE_RATE}_sr.wav")
    # The response's audio_content is binary.
    with open(target_wav_path, mode="wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)

    audio_meta = torchaudio.info(target_wav_path)
    duration = audio_meta.num_frames / audio_meta.sample_rate

    _jd = {
        "audio_filepath": target_wav_path,
        "speaker": f"spk_g_{_spk_index}",
        "duration": duration,
        "text": _text,
        "u_fid": u_fid,
    }

    return _jd


with open(manifest_path, encoding="utf-8", mode="w") as tm:
    with open(text_path, encoding="utf-8") as st:
        count = 0
        for t_index, line in enumerate(st):
            try:
                text = line.strip("\n").strip()

                # Set the text input to be synthesized
                synthesis_input = texttospeech.SynthesisInput(text=text)

                if GEN_ALL_TEXT:
                    for spk_index in range(len(SPEAKERS)):
                        env_index = random.choice(range(len(ENV)))
                        jd = get_synthesis_meta(text, spk_index, env_index)
                        json.dump(jd, tm)
                        tm.write("\n")
                        count = count + 1
                else:
                    spk_index = t_index % len(SPEAKERS)
                    env_index = t_index % len(ENV)
                    jd = get_synthesis_meta(text, spk_index, env_index)
                    json.dump(jd, tm)
                    tm.write("\n")
                    count = count + 1

                _log.info(f"[{count}/{t_index}]")

            except Exception as e:
                _log.error(e)
