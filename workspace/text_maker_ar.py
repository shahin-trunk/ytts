import re

from gruut import sentences
from nemo_text_processing.text_normalization import Normalizer

from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak


class Settings:
    SPACE = " "
    BLANK = ""
    DOT = "."
    DOUBLE_SPACE = SPACE + SPACE
    DOUBLE_DOT = DOT + DOT

    REM_CHARS = ['“', '”', '…', '_', '\u200b', '/', 'ـ', '{', '|', '}', '\xa0', '«', '\u200f', '\u202a',
                 '\u202b', '\u202c', '\u202e', '\u202f', '\u2067', '\u2069', '﴾', '﴿', '»', '’', '•',
                 '\ufeff', '\u200c', '\u200e', 'ۖ', 'ۗ', 'ۚ', 'ٰ', 'ٓ', '‘', '″', '·', '×', 'İ', 'ı', '\x01']

    PUNCTUATIONS = ['!', '؟', '-', '،', '؛', ':', '.']
    PUNCTUATIONS_STR = "".join(sorted(PUNCTUATIONS))

    REP_MAP = {'٠': '0', '١': '1', '٢': '2', '٣': '3', '٦': '6', '٧': '7', '٤': '4', '٥': '5', '٨': '8',
               '٩': '9', '٪': '%', '٫': '،', '—': '-', 'پ': 'ب', 'چ': 'ج', 'ڤ': 'ف', 'ﺋ': 'ئ', 'ﺤ': 'ح', 'ﺧ': 'خ',
               'ﻄ': 'ط', '٬': '،', 'ی': 'ى', 'ﻋ': 'ء', 'ﻖ': 'ق', 'ﻣ': 'م', 'ﻦ': 'ن', 'ﻮ': 'و', 'ﻲ': 'ي', 'ﻼ': 'لا',
               'ٓ': 'ا', '–': '-', 'ﺎ': 'ا', 'ﺑ': 'ب', 'ﺔ': 'ة', 'ﺘ': 'ت', 'ﺮ': 'ر', 'ﺳ': 'س', 'ﺸ': 'ش', 'ﻘ': 'ق',
               'ﻟ': 'ل', 'ﻧ': 'ن', 'ﻬ': 'ه', 'ﻳ': 'ي', 'ﻴ': 'ي', 'ﻵ': 'لآ', 'ﻻ': 'لا', '. . .': '.', '. .': '.',
               '،.': '،', 'ﷺ': SPACE + 'صلى الله عليه وسلم' + SPACE, '©': SPACE + 'copy-right' + SPACE}

    SYMBOLS_TO_RM_PREPR = ["\"", "\'", "“", "”", "«", "»", "(", ")", "<", ">"]
    SYMBOLS_TO_RM_PREPR_RE = re.compile(r'([' + ''.join(SYMBOLS_TO_RM_PREPR) + r'])')
    WS_RE = re.compile(r"\s+")

    TEXT_CASE = "cased"
    LANGUAGE = "ar"
    X01 = '\x01'
    PHONEME_SEPERATOR = ""
    DIACRITIZER_HOST = "77.242.240.151"
    DIACRITIZER_PORT = "5050"


class Diacritizer:

    def __init__(self) -> None:
        super().__init__()
        self.__diacritizer = FarasaDiacritizer()

    @staticmethod
    def __text_preprocess(text: str):
        text_filtered = Settings.SYMBOLS_TO_RM_PREPR_RE.sub(r'', text)
        return ' '.join(text_filtered.split())

    def diacritize(self, texts: list):
        return [self.__diacritizer.diacritize(text) for text in texts]


class Phonemizer:
    def __init__(self) -> None:
        super().__init__()
        self.__phonemizer = ESpeak(language="ar",
                                   backend="espeak-ng",
                                   punctuations=Settings.PUNCTUATIONS_STR,
                                   keep_puncs=True)

    def phonemize(self, texts: list):
        return [self.__phonemizer.phonemize(text, separator=Settings.PHONEME_SEPERATOR, language=Settings.LANGUAGE) for
                text in texts]


class Formatter:

    def __init__(self) -> None:
        super().__init__()
        self.__normalizer = Normalizer(input_case=Settings.TEXT_CASE, lang=Settings.LANGUAGE)

    def format(self, text) -> list:
        texts = []
        text = self.__normalizer.normalize(text, False, True, True)
        for sent in sentences(text, lang="ar"):
            sentence = sent.text.strip().replace(Settings.DOUBLE_SPACE, Settings.SPACE).strip()
            punct_in_end = False
            for ch in Settings.PUNCTUATIONS:
                sentence = sentence.replace(Settings.SPACE + ch, ch + Settings.SPACE)
                sentence = sentence.replace(ch + ch, ch + Settings.SPACE)
                sentence = sentence.replace(ch + ".", ch + Settings.SPACE)
                sentence = sentence.replace("." + ch, ch + Settings.SPACE)
                sentence = sentence.replace("-" + ch, ch + Settings.SPACE)
                sentence = sentence.replace(ch + "-", ch + Settings.SPACE)
                sentence = sentence.replace(Settings.DOUBLE_SPACE, Settings.SPACE)

                if sentence.startswith(ch):
                    sentence = sentence.replace(ch, Settings.SPACE, 1)
                    sentence = sentence.replace(Settings.DOUBLE_SPACE, Settings.SPACE).strip()

                if sentence.endswith(ch):
                    punct_in_end = True

            if not punct_in_end:
                sentence = sentence + Settings.DOT

            sentence = sentence.replace(Settings.X01, Settings.SPACE)
            sentence = sentence.replace(Settings.DOUBLE_SPACE, Settings.SPACE).strip()
            if sentence and len(sentence) > 0:
                texts.append(sentence)
        return texts


class Cleaner:

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def clean(text: str) -> str:
        text = text.strip("\n").strip("\t").strip()
        text = re.sub(Settings.WS_RE, " ", text).strip()
        text = text.replace("\n", Settings.SPACE).replace("\t", Settings.SPACE)

        for ch in Settings.REM_CHARS:
            text = text.replace(ch, Settings.SPACE)
            text = text.replace(Settings.DOUBLE_SPACE, Settings.SPACE)

        for ch_k, ch_v in Settings.REP_MAP.items():
            text = text.replace(ch_k, ch_v)
            text = text.replace(Settings.DOUBLE_SPACE, Settings.SPACE)

        for ch in Settings.PUNCTUATIONS:
            text = text.replace(Settings.SPACE + ch, ch + Settings.SPACE)
            text = text.replace(Settings.DOUBLE_SPACE, Settings.SPACE)
            text = text.replace(ch, ch + Settings.SPACE)
            text = text.replace(Settings.DOUBLE_SPACE, Settings.SPACE)
            text = text.strip().replace(ch + ch, ch)
            text = text.strip().replace(Settings.DOUBLE_SPACE, Settings.SPACE)

        for i in range(2):
            text = text.replace(Settings.DOUBLE_SPACE, Settings.SPACE)
            text = text.replace(Settings.DOUBLE_DOT, Settings.SPACE)
            text = text.replace(Settings.DOUBLE_SPACE, Settings.SPACE)

        return text


class TextMaker:

    def __init__(self) -> None:
        super().__init__()
        self.__cleaner = Cleaner()
        self.__formatter = Formatter()
        self.__diacritizer = Diacritizer()
        self.__phonemizer = Phonemizer()

    def make(self, text: str, phonemize: bool = False) -> list:
        text = self.__cleaner.clean(text)
        texts = self.__formatter.format(text)
        texts = self.__diacritizer.diacritize(texts)
        if phonemize:
            texts = self.__phonemizer.phonemize(texts)
        return texts
