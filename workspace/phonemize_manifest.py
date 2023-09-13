from gruut import sentences
from nemo_text_processing.text_normalization import Normalizer
from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak

PUNCTUATIONS = "".join(sorted(set([ch for ch in "'()-: ,.؟!،؛?"])))
phonemizer = ESpeak(language="ar", backend="espeak-ng", punctuations=PUNCTUATIONS, keep_puncs=True)

text = "وَأَحْصَتْ حُكُومَةُ الوِلايَةِ مئتَانْ وَثَمَانِيَةَ وَعِشْرُونَ شَخْصًاً آخَرِينَ أَصْبَحوا بِلا مَأْوًى، بَيْنَمَا تَمَّ إِجْلاءُ ثَلاثِ مِئَةٍ وَثَمَانِيَةٍ وَثَلاثُونَ شَخْصًاً مِنْ المِنْطَقَةِ السَّاحِليَّةِ شَمَالَ مَدِينَةِ سَاو بَاولو، فِي الوَقْتِ اَلَّذِي تَعْمَلُ فِيهِ طَوَاقِمِ الإِنْقَاذِ عَلَى مُسَاعَدَةِ المُتَضَرِّرِينَ مِنْ العَاصِفَة beethowen."

# text = "We are testing, you ،know? right. keiv,"
print(phonemizer.phonemize(text, separator="", language="ar"))
print("." == ".")
