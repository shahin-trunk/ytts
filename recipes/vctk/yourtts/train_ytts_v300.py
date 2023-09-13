import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig

torch.set_num_threads(24)

CHARACTERS = "".join(sorted(set([ch for ch in "ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىيًٌٍَُِّْ"])))
PUNCTUATIONS = "".join(sorted(set([ch for ch in "'()-: ,.؟!،؛?"])))
RUN_NAME = "YTTS-AR-IIAI"
EXP_ID = "v300_1"
SPK_EMBEDDING_ID = "v300_1"
DATA_ID = "v107"

RESTORE_PATH = "/data/asr/workspace/audio/tts/expmt/ytts/v300/YTTS-AR-IIAI-August-07-2023_12+35AM-452d4855/best_model.pth"
EXPMT_PATH = f"/data/asr/workspace/audio/tts/expmt/ytts/{EXP_ID}"

SKIP_TRAIN_EPOCH = False

BATCH_SIZE = 32
EVAL_BATCH_SIZE = 4
SAMPLE_RATE = 22050
MAX_AUDIO_LEN_IN_SECONDS = 20

BASE_PATH = "/data/asr/workspace/audio/tts"
NUM_RESAMPLE_THREADS = 10


def get_dataset(index: int = 1):
    return BaseDatasetConfig(
        formatter="iiai_tts",
        dataset_name=f"{index}",
        meta_file_train=f"data/audio/manifest/{DATA_ID}/manifest_{index}_{SAMPLE_RATE}sr.json",
        meta_file_val=f"data/audio/manifest/{DATA_ID}/manifest_{index}_{SAMPLE_RATE}sr_eval.json",
        path=BASE_PATH,
        language="ar",
    )


def get_gen_dataset(index: int = 1):
    return BaseDatasetConfig(
        formatter="iiai_tts",
        dataset_name=f"gen_{DATA_ID}_{index}",
        meta_file_train=f"data/audio/wav/gen_data/sentence_02_08_23/manifest/manifest_{index}_22050sr.json",
        meta_file_val=f"data/audio/wav/gen_data/sentence_02_08_23/manifest/manifest_{index}_22050sr_eval.json",
        path=BASE_PATH,
        language="ar",
    )


DATASETS_CONFIG_LIST = [get_dataset(ds_index) for ds_index in [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]]

for i in range(1, 6):
    DATASETS_CONFIG_LIST.append(get_gen_dataset(i))

SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "/data/asr/workspace/audio/tts/expmt/se/ar/v5/run-August-05-2023_06+21AM-452d4855/checkpoint_72000.pth"
)
SPEAKER_ENCODER_CONFIG_PATH = "/data/asr/workspace/audio/tts/expmt/se/ar/v5/run-August-05-2023_06+21AM-452d4855/config.json"

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
    d_vector_dim=1024,
    num_layers_text_encoder=12,
    num_layers_flow=40,
    num_layers_posterior_encoder=40,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    use_speaker_encoder_as_loss=True,
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
    save_n_checkpoints=2,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    use_phonemes=False,
    # phonemizer="espeak",
    # phoneme_language="ar",
    # compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="ar_bw_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=CHARACTERS,
        punctuations=PUNCTUATIONS,
        is_unique=True,
        is_sorted=True,
    ),
    precompute_num_workers=12,
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
            "مِن أَجْل تَأْمِين مُوَاطِنِيهَا الْمَوْجُودِين فِي السودَان.",
            "spk_123",
            None,
            "ar",
        ],
        [
            "اعْتَاد سُورِيون عَلَى وُجُود كُرْسِي فَارِغ حَوْل حَوْل طَاوِلَة الْإِفْطَار لِلْأَسَفْ.",
            "spk_428",
            None,
            "ar",
        ],
        [
            "بِالتَّعَاوُن مَع الْأَميرِكِي فِي قَرَار.",
            "spk_123",
            None,
            "ar",
        ],
        [
            "يَعْنِي هُنَاك مَسَاعِي مِن عَدْددُول الْحُصُول عَلَى عُضْوِيَّة الْبَرْكَس السُّعُودِيَّة وَاحِدَة مِن هَذِه الدوَل مَا هِي الِاسْتِفَادَة اَلَّتِي تَبْحَث عَنْهَا الريَاضْ.",
            "spk_64",
            None,
            "ar",
        ],
        [
            "واشِنْطُنْ تُعيدُ النَّظَرَ في جُهودِ الوَساطَةِ لِوَقْفِ القِتالِ في السّودانِ . في سِيَاقِ الأَزْمَةِ السّودانيَّةِ أَيْضًا . قالَتْ وِزارَةُ الخارِجيَّةِ الأَمْريكيَّةُ أَنَّ واشِنْطُنْ تُعيدُ النَّظَرَ مَعَ شُرَكائِها الأَفارِقَةِ والْعَرَبِ . في كَيْفيَّةِ المُضيِّ قُدُمًا في جُهودِ الوَساطَةِ في الصِّراعِ في السّودانِ . وَتَأْمَلُ في تَقْديمِ تَوْصياتٍ بِحُلولِ نِهايَةِ الأُسْبوعِ . وَأَشارَتْ الوِزارَةُ إِلَى أَنَّها تُجْري مُشاوَراتٍ مَعَ السُّعوديَّةِ والأَطْرافِ الأِفْريقيَّةِ والْعَرَبِ وَشُرَكاءَ آخَرينَ . بِشَأْنِ الطَّريقِ لِلْمُضيِّ قُدُمًا في حَلِّ الأَزْمَةِ السّودانيَّةِ . وَأَنَّها تَعْتَقِدُ أَنَّها سَتَخْرُجُ بِتَوْصياتٍ في الأَيّامِ المُقْبِلَة. وَكانَ المَسْؤولونَ الأَمْريكيّونَ والسُّعوديّونَ قَدْ حَذَّروا السَّبْتَ مِنْ أَنَّهُمْ قَدْ يوقِفونَ جُهودَ الوَساطَةِ هَذِه. في الوَقْتِ اَلَّذي تَمَّ فيهِ انْتِهاكُ أَكْثَرَ مِنْ هُدْنَةٍ بِشَأْنِ وَقْفِ إِطْلاقِ النّارِ مِنْ قِبَلِ ما أَسْمَوْهُمْ بِالْأَطْرافِ المُتَنازِعَةِ في السّودانِ.",
            "spk_428",
            None,
            "ar",
        ],
        [
            "واشِنْطُنْ تُعيدُ النَّظَرَ في جُهودِ الوَساطَةِ لِوَقْفِ القِتالِ في السّودانِ . في سِيَاقِ الأَزْمَةِ السّودانيَّةِ أَيْضًا . قالَتْ وِزارَةُ الخارِجيَّةِ الأَمْريكيَّةُ أَنَّ واشِنْطُنْ تُعيدُ النَّظَرَ مَعَ شُرَكائِها الأَفارِقَةِ والْعَرَبِ . في كَيْفيَّةِ المُضيِّ قُدُمًا في جُهودِ الوَساطَةِ في الصِّراعِ في السّودانِ . وَتَأْمَلُ في تَقْديمِ تَوْصياتٍ بِحُلولِ نِهايَةِ الأُسْبوعِ . وَأَشارَتْ الوِزارَةُ إِلَى أَنَّها تُجْري مُشاوَراتٍ مَعَ السُّعوديَّةِ والأَطْرافِ الأِفْريقيَّةِ والْعَرَبِ وَشُرَكاءَ آخَرينَ . بِشَأْنِ الطَّريقِ لِلْمُضيِّ قُدُمًا في حَلِّ الأَزْمَةِ السّودانيَّةِ . وَأَنَّها تَعْتَقِدُ أَنَّها سَتَخْرُجُ بِتَوْصياتٍ في الأَيّامِ المُقْبِلَة. وَكانَ المَسْؤولونَ الأَمْريكيّونَ والسُّعوديّونَ قَدْ حَذَّروا السَّبْتَ مِنْ أَنَّهُمْ قَدْ يوقِفونَ جُهودَ الوَساطَةِ هَذِه. في الوَقْتِ اَلَّذي تَمَّ فيهِ انْتِهاكُ أَكْثَرَ مِنْ هُدْنَةٍ بِشَأْنِ وَقْفِ إِطْلاقِ النّارِ مِنْ قِبَلِ ما أَسْمَوْهُمْ بِالْأَطْرافِ المُتَنازِعَةِ في السّودانِ.",
            "spk_g_0",
            None,
            "ar",
        ],
        [
            "واشِنْطُنْ تُعيدُ النَّظَرَ في جُهودِ الوَساطَةِ لِوَقْفِ القِتالِ في السّودانِ . في سِيَاقِ الأَزْمَةِ السّودانيَّةِ أَيْضًا . قالَتْ وِزارَةُ الخارِجيَّةِ الأَمْريكيَّةُ أَنَّ واشِنْطُنْ تُعيدُ النَّظَرَ مَعَ شُرَكائِها الأَفارِقَةِ والْعَرَبِ . في كَيْفيَّةِ المُضيِّ قُدُمًا في جُهودِ الوَساطَةِ في الصِّراعِ في السّودانِ . وَتَأْمَلُ في تَقْديمِ تَوْصياتٍ بِحُلولِ نِهايَةِ الأُسْبوعِ . وَأَشارَتْ الوِزارَةُ إِلَى أَنَّها تُجْري مُشاوَراتٍ مَعَ السُّعوديَّةِ والأَطْرافِ الأِفْريقيَّةِ والْعَرَبِ وَشُرَكاءَ آخَرينَ . بِشَأْنِ الطَّريقِ لِلْمُضيِّ قُدُمًا في حَلِّ الأَزْمَةِ السّودانيَّةِ . وَأَنَّها تَعْتَقِدُ أَنَّها سَتَخْرُجُ بِتَوْصياتٍ في الأَيّامِ المُقْبِلَة. وَكانَ المَسْؤولونَ الأَمْريكيّونَ والسُّعوديّونَ قَدْ حَذَّروا السَّبْتَ مِنْ أَنَّهُمْ قَدْ يوقِفونَ جُهودَ الوَساطَةِ هَذِه. في الوَقْتِ اَلَّذي تَمَّ فيهِ انْتِهاكُ أَكْثَرَ مِنْ هُدْنَةٍ بِشَأْنِ وَقْفِ إِطْلاقِ النّارِ مِنْ قِبَلِ ما أَسْمَوْهُمْ بِالْأَطْرافِ المُتَنازِعَةِ في السّودانِ.",
            "spk_64",
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
