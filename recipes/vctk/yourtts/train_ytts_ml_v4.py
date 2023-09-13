import json
import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.characters import _pad, _blank, _bos, _eos

torch.set_num_threads(24)

CHARACTERS = "".join(sorted(
    {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
     'x', 'z', 'æ', 'ç', 'ð', 'ħ', 'ŋ', 'ɐ', 'ɑ', 'ɒ', 'ɔ', 'ɕ', 'ɖ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɟ', 'ɡ', 'ɣ', 'ɨ', 'ɪ', 'ɬ',
     'ɭ', 'ɲ', 'ɳ', 'ɹ', 'ɾ', 'ʂ', 'ʃ', 'ʈ', 'ʊ', 'ʋ', 'ʌ', 'ʒ', 'ʔ', 'ʕ', 'ʰ', 'ʲ', 'ˈ', 'ˌ', 'ː', 'ˤ', '̃', '̩', '̪',
     'θ', 'χ', 'ᵻ'}))
PUNCTUATIONS = "".join(sorted({'!', '"', "'", ',', '-', '.', ':', ';', '?', '،', '؛', '؟'}))

RUN_NAME = "YTTS-AR-IIAI"
EXP_ID = "v4_ML"
REF_EXP_ID = "v3_ML"
SPK_EMBEDDING_VERSION = "v1"
PHN_CACHE_VERSION = "v1"
LNG_EMBEDDING_VERSION = "v1"
BASE_PATH = "/data/asr/workspace/audio/tts"
DATA_PATH = "data/audio/multi_lang"
EXPMT_PATH = os.path.join(BASE_PATH, f"expmt/ytts/{EXP_ID}")
REF_EXPMT_PATH = os.path.join(BASE_PATH, f"expmt/ytts/{REF_EXP_ID}")
PHN_CACHE_PATH = os.path.join(REF_EXPMT_PATH, f"phn_cache_{PHN_CACHE_VERSION}")
SPK_EMB_CACHE_PATH = os.path.join(REF_EXPMT_PATH, f"spk_emb_{SPK_EMBEDDING_VERSION}")
LNG_EMB_CACHE_PATH = os.path.join(REF_EXPMT_PATH, f"lng_emb_{LNG_EMBEDDING_VERSION}")
RESTORE_PATH = os.path.join(BASE_PATH,
                            "expmt/ytts/v1_ML/YTTS-AR-IIAI-August-23-2023_03+34PM-452d4855/best_model.pth")

os.makedirs(PHN_CACHE_PATH, exist_ok=True)
# os.makedirs(SPK_EMB_CACHE_PATH, exist_ok=True)
os.makedirs(LNG_EMB_CACHE_PATH, exist_ok=True)

LNG_EMB = {
    "hi": 0,
    "en": 1,
    "ar": 3,
    "ml": 4,
}

LNG_EMB_FILE = os.path.join(LNG_EMB_CACHE_PATH, "language_ids.json")
with open(LNG_EMB_FILE, mode="w") as lef:
    json.dump(LNG_EMB, lef)

SKIP_TRAIN_EPOCH = False
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 4
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
    get_dataset(manifest_train="manifest_gen_ar_ns_phoneme_ut_2000.json",
                manifest_eval="manifest_gen_ar_ns_phoneme_eval.json",
                d_name="gen",
                lang="ar"),
    # get_dataset(manifest_train="manifest_org_ar_ns_phoneme.json",
    #             manifest_eval="manifest_org_ar_ns_phoneme_eval.json",
    #             d_name="gen",
    #             lang="ar"),
    # get_dataset(manifest_train="manifest_org_en_ns_phoneme.json",
    #             manifest_eval="manifest_org_en_ns_phoneme_eval.json",
    #             d_name="org",
    #             lang="en"),
    get_dataset(manifest_train="manifest_gen_en_ns_phoneme_ut_1000.json",
                manifest_eval="manifest_gen_en_ns_phoneme_eval.json",
                d_name="gen",
                lang="en"),
    get_dataset(manifest_train="manifest_org_hi_ns_phoneme.json",
                manifest_eval="manifest_org_hi_ns_phoneme_eval.json",
                d_name="org",
                lang="hi"),
    get_dataset(manifest_train="manifest_org_ml_ns_phoneme.json",
                manifest_eval="manifest_org_ml_ns_phoneme_eval.json",
                d_name="org",
                lang="ml"),
]

SPEAKER_ENCODER_CHECKPOINT_PATH = (
    os.path.join(BASE_PATH, "expmt/se/multi/v8/run-August-22-2023_09+48AM-452d4855/checkpoint_137000.pth"))
SPEAKER_ENCODER_CONFIG_PATH = os.path.join(BASE_PATH,
                                           "expmt/se/multi/v8/run-August-22-2023_09+48AM-452d4855/config.json")

D_VECTOR_FILES = []

for dataset_conf in DATASETS_CONFIG_LIST:
    d_cache_dir = os.path.join(SPK_EMB_CACHE_PATH, f"{dataset_conf.dataset_name}")
    d_vector_cache_path = os.path.join(d_cache_dir, "d_vector_files.json")
    with open(d_vector_cache_path, encoding="utf-8") as dvcp:
        d_v_files = json.load(dvcp)

    d_v_files = [d_v_file if BASE_PATH in d_v_file else os.path.join(BASE_PATH, d_v_file) for d_v_file in d_v_files]
    D_VECTOR_FILES.extend(d_v_files)

print(f"D_VECTOR_FILES: {D_VECTOR_FILES}")

# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=2048,
    num_mels=400,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    use_sdp=False,
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=2048,
    out_channels=1025,
    num_heads_text_encoder=4,
    num_layers_text_encoder=12,
    num_layers_dp_flow=8,
    num_hidden_channels_dp=256,
    num_layers_flow=32,
    num_layers_posterior_encoder=32,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    use_speaker_encoder_as_loss=True,
    # use_language_embedding=True,
    # embedded_language_dim=8,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    epochs=10000,
    output_path=EXPMT_PATH,
    lr_gen=0.0003,
    lr_disc=0.0003,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=RUN_NAME,
    run_description=RUN_NAME,
    dashboard_logger="tensorboard",
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_loader_workers=16,
    print_step=100,
    plot_step=300,
    log_model_step=1000,
    save_step=10000,
    save_n_checkpoints=3,
    save_checkpoints=True,
    print_eval=True,
    # use_phonemes=False,
    # phonemizer="espeak",
    # phoneme_language="ar",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="ar_bw_cleaners",
    # use_language_embedding=True,
    # language_ids_file=LNG_EMB_FILE,
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad=_pad,
        eos=_eos,
        bos=_bos,
        blank=_blank,
        characters=CHARACTERS,
        punctuations=PUNCTUATIONS,
        is_unique=True,
        is_sorted=True,
    ),
    # precompute_num_workers=8,
    # phoneme_cache_path=PHN_CACHE_PATH,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    use_d_vector_file=True,
    d_vector_dim=1024,
    d_vector_file=D_VECTOR_FILES,
    test_sentences=[
        [
            'ʔattːˌadaχχˈulaːtˌi ˈaʕtamˌadt ʕˈalaː ʕˌamalˈiːaːtˌi ʔasssˈuːqi ʔalmaftˈuːħatˌi، ʔˈai tˈilka ʔalmˌuʕaːmˈalaːtˌu ʔˈallatˌiː tˈasmaħˌu lilbˌanki ʔalmarkˈaziːˌi bitˌaufˌiːɹi ʔˈau astˈinzaːfˌi ʔasssˌuiˈuːlatˌi qas̪ˈi.ːɹatˌi ʔalʔˈadʒalˌi muqˈaːbilˌa ʔaˈuɹaːqˌin maːlˈiːatˌin mˈin ʔalmuqɹˈidˤi.ːnˌa، ħˈasbamˌaː qˈaːla ʔˈaʃχaːs̪ˌun mˌut̪ːa.lˈiʕuːnˌa ʕˈalaː ʔalʔˈamr.',
            "ar_spk_1",
            None,
            "ar",
        ],
        [
            'wˈaʔaħs̪ˌa.t ħukˈuːmatˌu ʔalwilˈaːjatˌi mʔtˈaːn wˌaθamaːnˈiːatˌa waʕˈiʃɹuːnˌa ʃˈaχs̪aː ʔaːχˈaɹiːnˌa ʔˈas̪baħˌuː bilˌaː mˈaʔwa، baˌinamˌaː tˈamma ʔˈidʒlaːʔˌu θˈalaːθˌi mˈiʔatˌin wˌaθamaːnˈiːatˌin wˌaθalˈaːθuːnˌa ʃˈaχs̪aː mˈin ʔalmint̪ˈa.qatˌi ʔasssˌaːħilˈiːatˌi ʃˈamaːlˌa madˈiːnatˌi saːw baːwlˌuː، fiː ʔalwˈaqti ʔˈallaðˌiː tˈaʕmalˌu fiːhˌi t̪ˌa.u.ˈaːqimˌi ʔalʔˈinqaːðˌi ʕˈalaː mˌusaːʕˈadatˌi ʔalmˌutadˤa.rɹˈiɹiːnˌa mˈin ʔalʕaːs̪ˈi.fat.',
            "ar_spk_2",
            None,
            "ar",
        ],
        [
            'ħˈaððaɹˌa ɹˈaʔiːsˌu ʔalwuzˈaɹaːʔˌi ʔassswˈidiːˌu ʔˈuːlf kɹiːstrsˌuːn mˈin fas̪lˌi t̪ˈa.labˌi ʕudˤwˈiːatˌi bilˌaːdihˌi fiː ħˈilfi ʃˈamaːlˌi ʔalʔat̪lˈasiːˌi ʕˈan finlˌandaː، waðˈalikˌa baʕdˌa aʕtˈiɹaːfˌi ʔalħˈilfi liʔˌauːˌali mˈarɹatˌin biʔˌannahˌu qˈad jˌataʕaˈiːanˌu fas̪lˌu ʔat̪ːˌa.labˈainˌi ʕˈalaː χalfˈiːatˌi aʕtˈiɹaːdˤˌi. tˈurkiːˌaː.',
            "ar_spk_3",
            None,
            "ar",
        ],
        [
            'qˈaːla ʔarrɹˈaʔiːsˌu ʔattːanfˈiːðiːˌu liʔˌaħadˌi ʔˈakbaɹˌi ʔalbunˌuːki ʔalʕaːmˈilatˌi fiː ʔasssˌuʕuːdˈiːatˌi ʔˈinna dʒˈuhuːdˌa ʔalmamlˈakatˌi litˌaχfiːfˌi ʔˈazmatˌi ʔasssˌuiˈuːlatˌi ʔalʔaχˈiːɹatˌi ʔˈallatˌiː ʃahˈidahˌaː ʔannnˈiða.ːmˌu ʔalmˈaːliːˌu badˌaʔat tˈuʔtiː θimˈaːɹahˌaː watˈaʕmalˌu ʕˈalaː tahdˈiʔatˌi ʔalʔˈaswaːq.',
            "ar_spk_4",
            None,
            "ar",
        ],
        [
            'jˈaʕmalˌu ʔalmˈadʒlisˌu ʕˈalaː fadˤdˤˌi. ʔalmˌunaːzˈaʕaːtˌi wˌataswiːˈatihˌaː wˌaʔidaːɹˈatihˌaː dˈaːχilˌa alˌiaːtːˈiħaːdˌi ʔalʔifɹˈiːqiːˌi، wˌaiusˈaːʕidˌu fiː ʔattːˈaħdˤi.ːɹˌi litˌanði.ːmˌi alˌiaːntiχˈaːbaːtˌi walʔˈiʃɹaːfˌi ʕalˈaihˌaː fiː ʔaddːˈuːalˌi ʔalʔˈaʕdˤa.ːʔˌi، wˌaiˈahdifˌu ʔˈilaː tˈaʕziːzˌi ʔasssˈalaːmˌi walʔˈamni fiː ʔifɹˈiːqiˌaː.',
            "ar_spk_5",
            None,
            "ar",
        ],
        [
            'ʔˈat̪laqt kuːrjˌaː ʔaʃʃˌamaːlˈiːatˌi، ʔaljˈaumˌa s̪ˈa.ːɹuːχˌaː baːlˌistiːˌaː ɣˈaiɹˌa muħˈadːadˌin biˌaːtːidʒˌaːhi baħɹˌi ʔaljˈaːbaːnˌi، hˈuːa ʔaθθθˈaːniː fiː ʔˈaqallˌa mˈin θˌamaːnˈiːatˌi wˌaʔarbˈaʕuːnˌa saːʕˌatan، waˈiaʔtˌiː ɣˈadaːtˌa tadɹˈiːbaːtˌin mˌuʃtaɹˈakatˌin baˌina dʒaˈiʃaˌi ʔalwilˈaːjaːtˌi ʔalmˌutːaħˈidatˌi wˈakuːrjˌaː ʔaldʒˌanuːbˈiːatˌi، wˈafqa mˈaː ʔˈaʕlanˌat ɹiʔˈaːsatˌu ʔalʔˈarkaːnˌi ʔalmˌuʃtaɹˈakatˌi fiː siːwl.',
            "ar_spk_6",
            None,
            "ar",
        ],
        [
            'waːʃˌint̪u.n tˈuʕiːdˌu ʔannnˈaða.ɹˌa fˈiː dʒˈuhuːdˌi ʔalwasˈaːt̪a.tˌi liwˌaqfi ʔalqˈitaːlˌi fˈiː ʔassswdˈaːni . fˈiː siːˌaːqi ʔalʔˈazmatˌi ʔassswdaːnˈiːatˌi ʔˈaidˤˌa.n . qˈaːlat wizˈaːɹatˌu ʔalχˌaːɹidʒˈiːatˌi ʔalʔˌamɹiːkˈiːatˌu ʔˈanna waːʃˌint̪u.n tˈuʕiːdˌu ʔannnˈaða.ɹˌa mˈaʕa ʃˌuɹakˈaːʔihˌaː ʔalʔˌafaːɹˈiqatˌi waːlʕˌaɹabˌi . fˈiː kaˌifiːˌati ʔalmˈudˤiːˌi qˈudumˌan fˈiː dʒˈuhuːdˌi ʔalwasˈaːt̪a.tˌi fˈiː ʔas̪s̪ːˈi.ɹaːʕˌi fˈiː ʔassswdˈaːni . watˈaʔmalˌu fˈiː tˈaqdiːmˌi tˌaus̪ˈijjaːtˌin biħˌuluːlˌi nihˈaːjatˌi ʔalʔˈusbuːʕˌi . waʔˈaʃaːɹˌat ʔalwizˈaːɹatˌu ʔˈilaː ʔˈannahˌaː tˈudʒɹiː mˌuʃaːwˈaɹaːtˌin mˈaʕa ʔasssˌuʕuːdˈiːatˌi walʔˈat̪ɹaːfˌi ʔalʔˌifɹiːqˈiːatˌi waːlʕˌaɹabˌi wˌaʃuɹˈakaːʔˌa ʔaːχˈaɹiːnˌa . biʃˌaʔni ʔat̪ːˈa.ɹiːqˌi lilmˌudˤiːˌi qˈudumˌan fˈiː ħˈalli ʔalʔˈazmatˌi ʔassswdaːnˈiːatˌi . waʔˈannahˌaː taʕtˈaqidˌu ʔˈannahˌaː satˌaχɹudʒˌu bitˌaus̪ˈijjaːtˌin fˈiː ʔalʔaˈiːaːmˌi ʔalmuqbˈilat. wˈakaːnˌa ʔalmasʔˈuːluːnˌa ʔalʔamɹˈiːkiːwnˌa wasssuʕˈuːdiːwnˌa qˈad ħˈaððaɹˌuː ʔasssˈabta mˈin ʔˈannahˌum qˈad jwqˈifuːnˌa dʒˈuhuːdˌa ʔalwasˈaːt̪a.tˌi hˈaðih. fˈiː ʔalwˈaqti ʔˈallaðˌiː tˈamma fiːhˌi antˈihaːkˌu ʔˈakθaɹˌa mˈin hˈudnatˌin biʃˌaʔni wˈaqfi ʔˈit̪laːqˌi ʔannnˈaːɹi mˈin qˈibalˌi mˈaː ʔasmˈauhˌum biˌaːlʔat̪ɹˌaːfi ʔalmˌutanaːzˈiʕatˌi fˈiː ʔassswdˈaːni.',
            "ar_spk_7",
            None,
            "ar",
        ],
        [
            'ɪt tˈʊk mˌiː kwˈaɪt ɐ lˈɔŋ tˈaɪm tə dɪvˈɛləp ɐ vˈɔɪs, ænd nˈaʊ ðæt aɪ hæv ɪɾ aɪm nˌɑːt ɡˌoʊɪŋ təbi sˈaɪlənt.',
            "en_spk_1",
            None,
            "en",
        ],
        [
            'pɹˈaɪɚ tə noʊvˈɛmbɚ twˈɛnti twˈɛnti θɹˈiː.',
            "en_spk_109",
            None,
            "en",
        ],
        [
            'bˈʌr.eː bˈʌr.eː dˈeːʃõ mẽː ˈɛːsi cʰˈoːʈi cʰˈoːʈi bˈaːtẽː hˈoːti ɾˈʌhətˌi hɛː',
            "hi_spk_1",
            None,
            "hi",
        ],
        [
            'nˈaːɖinoːʈ pɾˈɐdibˌɐddʰədɐ uɳɖˈaːjaːl mˈaːtɾəmeː vˈiɡəsˌɐnəm uɳɖˈaːɡuɡˌɐjuɭɭˌuːvennɨ mˈukʰjəmˌɐntɾi pˈiɳərˌaːji vˈiɟəjən',
            "ml_spk_1",
            None,
            "ml",
        ],
        [
            'cˈesɨ lˈoːɡəɡˌɐpːɨ kiɾˈiːɖəm lˈoːɡɐ onnˈaːm nˈɐmbəɾ tˈaːɾəvum mˈun lˈoːɡəcˌaːmpjənˌumaːjɐ nˈoːɾvejˌuɖe mˈaːɡnəsɨ kˈaːɭsənɨ',
            "ml_spk_5",
            None,
            "ml",
        ],
        [
            'vˈiɕvənˌaːtʰən ˈaːnəndˌinuɕˌeːʂəm cˈesɨ lˈoːɡəɡˌɐpːɨ pʰˈaɪnəlˌɐlilˌetːunnɐ ˈaːdjɐ ˈintjən tˈaːɾəmˌennɐ nˈeːʈːəm nˈeːɾətːeː tˈɐnne pɾɐɡnˈaːnəndɐ svˈɐndəmˌaːkːijˌiɾunnu.',
            "ml_spk_19",
            None,
            "ml",
        ],
    ],
    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=9.0,
    save_all_best=True
)

# Load all the datasets samples and split traning and evaluation sets
for dataset in config.datasets:
    print(dataset)

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
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=EXPMT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
