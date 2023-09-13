import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.characters import _pad, _blank, _bos, _eos

torch.set_num_threads(24)

# ASR DIACRITICS CHARSET

CHARACTERS = "".join(sorted(set([ch for ch in "ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىيًٌٍَُِّْ"])))
CHARACTERS_PHN = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻʘɓǀɗǃʄǂɠǁʛpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟˈˌːˑʍwɥʜʢʡɕʑɺɧʲɚ˞ɫˤ̪"
PUNCTUATIONS = "".join(sorted(set([ch for ch in "'()-: ,.؟!،؛?"])))
RUN_NAME = "YTTS-AR-IIAI"
EXP_ID = "v301_PHN"
SPK_EMBEDDING_ID = "v301_PHN_v4"
PHN_CACHE_VERSION = "v4"
DATA_ID = "v107"
BASE_PATH = "/data/asr/workspace/audio/tts"
RESTORE_PATH = os.path.join(BASE_PATH, "expmt/ytts/v301_1/YTTS-AR-IIAI-August-09-2023_06+14PM-452d4855/best_model.pth")
EXPMT_PATH = os.path.join(BASE_PATH, f"expmt/ytts/{EXP_ID}")
PHN_CACHE_PATH = os.path.join(EXPMT_PATH, f"phn_cache_{PHN_CACHE_VERSION}")

os.makedirs(PHN_CACHE_PATH, exist_ok=True)

SKIP_TRAIN_EPOCH = False

BATCH_SIZE = 32
EVAL_BATCH_SIZE = 4
SAMPLE_RATE = 22050
MAX_AUDIO_LEN_IN_SECONDS = 16

NUM_RESAMPLE_THREADS = 10


def get_dataset(index: int = 1, data_id: str = None):
    return BaseDatasetConfig(
        formatter="iiai_tts",
        dataset_name=f"{index}_{data_id}",
        meta_file_train=f"data/audio/manifest/{data_id}/manifest_{index}_{SAMPLE_RATE}sr.json",
        meta_file_val=f"data/audio/manifest/{data_id}/manifest_{index}_{SAMPLE_RATE}sr_eval.json",
        path=BASE_PATH,
        language="ar",
    )


def get_gen_dataset(index: int = 1):
    return BaseDatasetConfig(
        formatter="iiai_tts",
        dataset_name=f"gen_data_gg",
        meta_file_train=f"data/audio/wav/gen_data/sentence_02_08_23/manifest/manifest_combined_gg_speakers.json",
        meta_file_val=f"data/audio/wav/gen_data/sentence_02_08_23/manifest/manifest_combined_gg_speakers_eval.json",
        path=BASE_PATH,
        language="ar",
    )


DATASETS_CONFIG_LIST = [get_dataset(ds_index, "v107") for ds_index in [1, 2, 3, 4, 5, 6, 7, 8]]

DATASETS_CONFIG_LIST.append(get_gen_dataset())
DATASETS_CONFIG_LIST.extend([get_dataset(ds_index, "v302") for ds_index in [1, 2]])

print(f">>>DATASETS_CONFIG_LIST: \n{DATASETS_CONFIG_LIST}\n")

SPEAKER_ENCODER_CHECKPOINT_PATH = (
    os.path.join(BASE_PATH, "expmt/se/ar/v7/run-August-10-2023_10+01AM-452d4855/checkpoint_60000.pth"))
SPEAKER_ENCODER_CONFIG_PATH = os.path.join(BASE_PATH, "expmt/se/ar/v7/run-August-10-2023_10+01AM-452d4855/config.json")

D_VECTOR_FILES = []

for dataset_conf in DATASETS_CONFIG_LIST:
    embeddings_file = os.path.join(dataset_conf.path, f"speakers_{SPK_EMBEDDING_ID}_{dataset_conf.dataset_name}.pth")
    if not os.path.isfile(embeddings_file):
        print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            config_dataset_path=None,
            formatter_name=dataset_conf.formatter,
            dataset_name=dataset_conf.dataset_name,
            dataset_path=dataset_conf.path,
            meta_file_train=dataset_conf.meta_file_train,
            meta_file_val=dataset_conf.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    D_VECTOR_FILES.append(embeddings_file)

# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    num_mels=240,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    use_sdp=True,
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=2048,
    num_layers_text_encoder=12,
    num_layers_flow=40,
    num_layers_dp_flow=8,
    num_layers_posterior_encoder=40,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    use_speaker_encoder_as_loss=False,
    # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the
    # ResNet blocks type 1 like the VITS model Useful parameters to enable the Speaker Consistency Loss (SCL)
    # described in the paper
    # Useful parameters to enable multilingual training
    # use_language_embedding=True,
    # embedded_language_dim=4,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    epochs=10000,
    output_path=EXPMT_PATH,
    lr=0.001,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=RUN_NAME,
    run_description=RUN_NAME,
    dashboard_logger="tensorboard",
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_loader_workers=8,
    print_step=100,
    plot_step=300,
    log_model_step=1000,
    save_step=10000,
    save_n_checkpoints=3,
    save_checkpoints=True,
    print_eval=True,
    use_phonemes=True,
    phonemizer="espeak",
    phoneme_language="ar",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="ar_bw_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad=_pad,
        eos=_eos,
        bos=_bos,
        blank=_blank,
        characters=CHARACTERS,
        punctuations=PUNCTUATIONS,
        phonemes=CHARACTERS_PHN,
        is_unique=True,
        is_sorted=True,
    ),
    # characters=CharactersConfig(
    #     characters_class="TTS.tts.utils.text.characters.IPAPhonemes",
    #     pad=_pad,
    #     eos=_eos,
    #     bos=_bos,
    #     blank=_blank,
    #     characters=CHARACTERS_PHN,
    #     punctuations=PUNCTUATIONS,
    #     is_unique=True,
    #     is_sorted=True,
    # ),
    precompute_num_workers=8,
    phoneme_cache_path=PHN_CACHE_PATH,
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
            "التَّدَخُّلاتِ اعْتَمَدتْ عَلَى عَمَليَّاتِ السُّوقِ المَفْتُوحَةِ، أَيْ تِلْكَ المُعَامَلاتُ اَلَّتِي تَسْمَحُ لِلْبَنْكِ المَرْكَزِيِّ بِتَوْفِيرِ أَوْ اسْتِنْزَافِ السُّيُولَةِ قَصِيرَةِ الأَجَلِ مُقَابِلَ أَوْرَاقٍ مَاليَّةٍ مِنْ المُقْرِضِينَ، حَسْبَمَا قَالَ أَشْخَاصٌ مُطَّلِعُونَ عَلَى الأَمْرْ.",
            "spk_55247",
            None,
            "ar",
        ],
        [
            "وَأَحْصَتْ حُكُومَةُ الوِلايَةِ مئتَانْ وَثَمَانِيَةَ وَعِشْرُونَ شَخْصًاً آخَرِينَ أَصْبَحوا بِلا مَأْوًى، بَيْنَمَا تَمَّ إِجْلاءُ ثَلاثِ مِئَةٍ وَثَمَانِيَةٍ وَثَلاثُونَ شَخْصًاً مِنْ المِنْطَقَةِ السَّاحِليَّةِ شَمَالَ مَدِينَةِ سَاو بَاولو، فِي الوَقْتِ اَلَّذِي تَعْمَلُ فِيهِ طَوَاقِمِ الإِنْقَاذِ عَلَى مُسَاعَدَةِ المُتَضَرِّرِينَ مِنْ العَاصِفَة.",
            "spk_41941",
            None,
            "ar",
        ],
        [
            "حَذَّرَ رَئِيسُ الوُزَرَاءِ السّويدِيُّ أولْفْ كرِيسترسُونْ مِنْ فَصْلِ طَلَبِ عُضْوِيَّةِ بِلادِهِ فِي حِلْفِ شَمَالِ الأَطْلَسِيِّ عَنْ فِنْلَنْدَا، وَذَلِكَ بَعْدَ اعْتِرَافِ الحِلْفِ لِأَوَّلِ مَرَّةٍ بِأَنَّهُ قَدْ يَتَعَيَّنُ فَصْلُ الطَّلَبَيْنِ عَلَى خَلْفِيَّةِ اعْتِرَاضِ تُرْكيا.",
            "spk_53236",
            None,
            "ar",
        ],
        [
            "قَالَ الرَّئِيسُ التَّنْفِيذِيُّ لِأَحَدِ أَكْبَرِ البُنُوكِ العَامِلَةِ فِي السُّعُودِيَّةِ إِنَّ جُهُودَ المَمْلَكَةِ لِتَخْفِيفِ أَزْمَةِ السُّيُولَةِ الأَخِيرَةِ اَلَّتِي شَهِدَهَا النِّظَامُ المَاليُّ بَدَأَتْ تُؤْتِي ثِمَارَهَا وَتَعْمَلُ عَلَى تَهْدِئَةِ الأَسْوَاقْ.",
            "spk_64",
            None,
            "ar",
        ],
        [
            "يَعْمَلُ المَجْلِسُ عَلَى فَضِّ المُنَازَعَاتِ وَتَسْوِيَتِهَا وَإِدَارَتِهَا دَاخِلَ الِاتِّحَادِ الإِفْرِيقِيِّ، وَيُسَاعِدُ فِي التَّحْضِيرِ لِتَنْظِيمِ الِانْتِخَابَاتِ وَالْإِشْرَافِ عَلَيْهَا فِي الدُّوَلِ الأَعْضَاءِ، وَيَهْدِفُ إِلَى تَعْزِيزِ السَّلامِ وَالْأَمْنِ فِي إِفْرِيقيا.",
            "spk_428",
            None,
            "ar",
        ],
        [
            "أَطْلَقتْ كُورْيَا الشَّمَاليَّةِ، اليَوْمَ صَارُوخًاً بَالِسْتِيًّاً غَيْرَ مُحَدَّدٍ بِاتِّجَاهِ بَحْرِ اليَابَانِ، هُوَ الثَّانِي فِي أَقَلَّ مِنْ ثَمَانِيَةِ وَأَرْبَعُونَ سَاعَةً، وَيَأْتِي غَدَاةَ تَدْرِيبَاتٍ مُشْتَرَكَةٍ بَيْنَ جَيْشَيْ الوِلايَاتِ المُتَّحِدَةِ وَكُورْيَا الجَنُوبِيَّةِ، وَفْقَ مَا أَعْلَنَتْ رِئَاسَةُ الأَرْكَانِ المُشْتَرَكَةِ فِي سيولْ.",
            "spk_60207",
            None,
            "ar",
        ],
        [
            "واشِنْطُنْ تُعيدُ النَّظَرَ في جُهودِ الوَساطَةِ لِوَقْفِ القِتالِ في السّودانِ . في سِيَاقِ الأَزْمَةِ السّودانيَّةِ أَيْضًا . قالَتْ وِزارَةُ الخارِجيَّةِ الأَمْريكيَّةُ أَنَّ واشِنْطُنْ تُعيدُ النَّظَرَ مَعَ شُرَكائِها الأَفارِقَةِ والْعَرَبِ . في كَيْفيَّةِ المُضيِّ قُدُمًا في جُهودِ الوَساطَةِ في الصِّراعِ في السّودانِ . وَتَأْمَلُ في تَقْديمِ تَوْصياتٍ بِحُلولِ نِهايَةِ الأُسْبوعِ . وَأَشارَتْ الوِزارَةُ إِلَى أَنَّها تُجْري مُشاوَراتٍ مَعَ السُّعوديَّةِ والأَطْرافِ الأِفْريقيَّةِ والْعَرَبِ وَشُرَكاءَ آخَرينَ . بِشَأْنِ الطَّريقِ لِلْمُضيِّ قُدُمًا في حَلِّ الأَزْمَةِ السّودانيَّةِ . وَأَنَّها تَعْتَقِدُ أَنَّها سَتَخْرُجُ بِتَوْصياتٍ في الأَيّامِ المُقْبِلَة. وَكانَ المَسْؤولونَ الأَمْريكيّونَ والسُّعوديّونَ قَدْ حَذَّروا السَّبْتَ مِنْ أَنَّهُمْ قَدْ يوقِفونَ جُهودَ الوَساطَةِ هَذِه. في الوَقْتِ اَلَّذي تَمَّ فيهِ انْتِهاكُ أَكْثَرَ مِنْ هُدْنَةٍ بِشَأْنِ وَقْفِ إِطْلاقِ النّارِ مِنْ قِبَلِ ما أَسْمَوْهُمْ بِالْأَطْرافِ المُتَنازِعَةِ في السّودانِ.",
            "spk_28962",
            None,
            "ar",
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
