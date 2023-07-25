from TTS.tts.utils.text.phonemizers import get_phonemizer_by_name

phonemizer = get_phonemizer_by_name('espeak', **{"language": 'ar'})
print(phonemizer.supported_languages())
print(phonemizer.language)
