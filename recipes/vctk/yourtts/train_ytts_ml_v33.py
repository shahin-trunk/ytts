import json
import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig

torch.set_num_threads(24)

CHARACTERS_PHN = "".join(sorted(
    {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
     'x', 'z', 'æ', 'ç', 'ð', 'ħ', 'ŋ', 'ɐ', 'ɑ', 'ɒ', 'ɔ', 'ɕ', 'ɖ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɟ', 'ɡ', 'ɣ', 'ɨ', 'ɪ', 'ɬ',
     'ɭ', 'ɲ', 'ɳ', 'ɹ', 'ɾ', 'ʂ', 'ʃ', 'ʈ', 'ʊ', 'ʋ', 'ʌ', 'ʒ', 'ʔ', 'ʕ', 'ʰ', 'ʲ', 'ˈ', 'ˌ', 'ː', 'ˤ', '̃', '̩', '̪',
     'θ', 'χ', 'ᵻ'}))

CHARACTERS = "".join(
    sorted(
        {' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
         'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
         'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '£', 'Â', 'à', 'á', 'â', 'ã', 'ç', 'è', 'é',
         'ê', 'í', 'î', 'ñ', 'ó', 'ú', 'û', 'ā', 'ł', 'Š', 'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث',
         'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن',
         'ه', 'و', 'ى', 'ي', 'ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ'}))

PUNCTUATIONS = "".join(sorted({'!', '"', "'", '[', ']', '(', ')', ',', '-', '.', ':', ';', '?', '،', '؛', '؟'}))

RUN_NAME = "YTTS-AR-IIAI"
EXP_ID = "v33_ML"
REF_EXP_ID = "v34_ML"
SPK_EMBEDDING_VERSION = "v1"
PHN_CACHE_VERSION = "v1"
LNG_EMBEDDING_VERSION = "v1"
BASE_PATH = "/data/asr/workspace/audio/tts"
DATA_PATH = "data/audio/multi_lang/sel"
EXPMT_PATH = os.path.join(BASE_PATH, f"expmt/ytts/{EXP_ID}")
REF_EXPMT_PATH = os.path.join(BASE_PATH, f"expmt/ytts/{REF_EXP_ID}")
PHN_CACHE_PATH = os.path.join(REF_EXPMT_PATH, f"phn_cache_{PHN_CACHE_VERSION}")
SPK_EMB_CACHE_PATH = os.path.join(REF_EXPMT_PATH, f"spk_emb_{SPK_EMBEDDING_VERSION}")
LNG_EMB_CACHE_PATH = os.path.join(REF_EXPMT_PATH, f"lng_emb_{LNG_EMBEDDING_VERSION}")
RESTORE_PATH = os.path.join(BASE_PATH, "expmt/ytts/v109/YTTS-AR-IIAI-August-05-2023_10+46PM-0000000/best_model.pth")

os.makedirs(PHN_CACHE_PATH, exist_ok=True)
# os.makedirs(SPK_EMB_CACHE_PATH, exist_ok=True)
os.makedirs(LNG_EMB_CACHE_PATH, exist_ok=True)

LNG_EMB = {
    "ar": 0,
    "en": 1,
}

LNG_EMB_FILE = os.path.join(LNG_EMB_CACHE_PATH, "language_ids.json")
with open(LNG_EMB_FILE, mode="w") as lef:
    json.dump(LNG_EMB, lef)

SKIP_TRAIN_EPOCH = False
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
SAMPLE_RATE = 22050
MAX_AUDIO_LEN_IN_SECONDS = 16
NUM_RESAMPLE_THREADS = 10


def get_dataset(manifest_train: str, manifest_eval: str, d_name: str, lang: str = "ar"):
    return BaseDatasetConfig(
        formatter="iiai_tts",
        dataset_name=f"{lang}_{d_name}",
        meta_file_train=os.path.join(DATA_PATH, manifest_train),
        meta_file_val=os.path.join(DATA_PATH, manifest_eval),
        path=BASE_PATH,
        language=lang,
    )


DATASETS_CONFIG_LIST = [
    get_dataset(manifest_train="manifest_ar_ns.json",
                manifest_eval="manifest_ar_ns_eval.json",
                d_name="cmb",
                lang="ar"),
    get_dataset(manifest_train="manifest_en_ns.json",
                manifest_eval="manifest_en_ns_eval.json",
                d_name="cmb",
                lang="en"),
]

SPEAKER_ENCODER_CHECKPOINT_PATH = os.path.join(BASE_PATH,
                                               "expmt/se/multi/v9/run-August-30-2023_09+59PM-452d4855/checkpoint_24000.pth")
SPEAKER_ENCODER_CONFIG_PATH = os.path.join(BASE_PATH,
                                           "expmt/se/multi/v9/run-August-30-2023_09+59PM-452d4855/config.json")

D_VECTOR_FILES = []

for dataset_conf in DATASETS_CONFIG_LIST:
    d_cache_dir = os.path.join(SPK_EMB_CACHE_PATH, f"{dataset_conf.dataset_name}")
    d_vector_cache_path = os.path.join(d_cache_dir, "d_vector_files.json")
    with open(d_vector_cache_path, encoding="utf-8") as dvcp:
        d_v_files = json.load(dvcp)

    d_v_files = [d_v_file if BASE_PATH in d_v_file else os.path.join(BASE_PATH, d_v_file) for d_v_file in d_v_files]
    D_VECTOR_FILES.extend(d_v_files)

print(f" > D_VECTOR_FILES: {D_VECTOR_FILES}")

# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=2048,
    fft_size=2048,
    num_mels=320,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    use_sdp=False,
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=2560,
    out_channels=1025,
    num_heads_text_encoder=4,
    num_layers_text_encoder=12,
    num_layers_dp_flow=8,
    num_hidden_channels_dp=512,
    num_layers_flow=36,
    num_layers_posterior_encoder=36,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    use_speaker_encoder_as_loss=True,
    # encoder_sample_rate=44100,
    # use_language_embedding=True,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    epochs=10000,
    output_path=EXPMT_PATH,
    lr_gen=0.00023,
    lr_disc=0.00023,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=RUN_NAME,
    run_description=RUN_NAME,
    dashboard_logger="tensorboard",
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_loader_workers=24,
    print_step=1000,
    plot_step=1000,
    log_model_step=1000,
    save_step=10000,
    save_all_best=True,
    save_n_checkpoints=3,
    save_checkpoints=True,
    print_eval=True,
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="ar_bw_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="~",
        eos=">",
        bos="<",
        blank="^",
        characters=CHARACTERS,
        punctuations=PUNCTUATIONS,
        is_unique=True,
        is_sorted=True,
    ),
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    use_d_vector_file=True,
    d_vector_dim=2560,
    d_vector_file=D_VECTOR_FILES,
    # use_language_embedding=True,
    # language_ids_file=LNG_EMB_FILE,
    speaker_encoder_loss_alpha=9.0,
    use_language_weighted_sampler=False,
    use_speaker_weighted_sampler=True,
    use_length_weighted_sampler=True,
    use_weighted_sampler=True,
    weighted_sampler_attrs={
        "speaker_name": 1.0
    },
    weighted_sampler_multipliers={
        "speaker_name": {}
    },
    test_sentences=[
        [
            "مِن أَجْل تَأْمِين مُوَاطِنِيهَا الْمَوْجُودِين فِي السودَان.",
            "ar_spk_1",
            None,
            "ar",
        ],
        [
            "اعْتَاد سُورِيون عَلَى وُجُود كُرْسِي فَارِغ حَوْل حَوْل طَاوِلَة الْإِفْطَار لِلْأَسَفْ.",
            "ar_spk_11",
            None,
            "ar",
        ],
        [
            "يَعْنِي هُنَاك مَسَاعِي مِن عَدْددُول الْحُصُول عَلَى عُضْوِيَّة الْبَرْكَس السُّعُودِيَّة وَاحِدَة مِن هَذِه الدوَل مَا هِي الِاسْتِفَادَة اَلَّتِي تَبْحَث عَنْهَا الريَاضْ.",
            "ar_spk_2",
            None,
            "ar",
        ],
        [
            "عَلَى الْمُسْتَوَى الأمْنِي وَالْعَسْكَرِي وَالنَّفْسِي وَالسيَاسِي اَلَّذِي حَاكَتْه أَمْريْكَا وَحُلَفَائِهَا وَلا وَأَنَا لا نَنْسَى تَمَامًا هِي الْكَبنْبَالَات بِمَا يُسَمَّى بِأَصْدِقَاء سُوريا.",
            "ar_spk_22",
            None,
            "ar",
        ],
        [
            "I’ve been living a nightmare since Friday, knowing my baby is out there somewhere scared, or might be injured.",
            "en_spk_1",
            None,
            "ar",
        ],
        [
            "You can fool all of the people some of the time, and some of the people all of the time, but you can't fool all of the people all of the time.",
            "en_spk_21",
            None,
            "ar",
        ],
        [
            'Ukrainian President Volodymyr Zelensky says it may be possible to hold elections in Ukraine next year as scheduled, but the country would need financial support for such a complex undertaking during wartime.',
            "en_spk_31",
            None,
            "ar",
        ],
        [
            "Life is like a box of chocolates. You never know what you’re gonna get.",
            "en_spk_41",
            None,
            "en",
        ],
        [
            'I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character.',
            "en_spk_51",
            None,
            "en",
        ],
    ],
)

# Load all the datasets samples and split training and evaluation sets
# for dataset in config.datasets:
#     print(dataset)

train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Init the model
model = Vits.init_from_config(config)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH if RESTORE_PATH else "", skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=EXPMT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
