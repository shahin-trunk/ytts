import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig

threads = 8
workers = 4

torch.set_num_threads(threads)

# pylint: disable=W0105
# Name of the run for the Trainer
RUN_NAME = "YTTS-AR-IIAI"
EXP_ID = "v73"

# CHARACTERS and PUNCTUATIONS
CHAR_SET = {' ', 'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س',
            'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ـ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي', 'ً', 'ٌ', 'ٍ',
            'َ', 'ُ', 'ِ', 'ّ', 'ْ'}

CHAR_SET_NEW = {'ه', 'إ', 'ي', 'ق', 'ّ', 'س', 'ع', 'ئ', 'ن', 'ث', 'ف', 'ة', 'ت', 'ً', 'ٍ', 'و', 'ب', 'ص', 'ى', 'د', 'غ',
                'ذ', 'ك', 'ج', 'َ', 'م', 'خ', ' ', 'ٌ', 'ء', 'ش', 'ؤ', 'ظ', 'ر', 'ل', 'ز', 'ُ', 'آ', 'ِ', 'ا', 'ح', 'ْ',
                'ط', 'أ', 'ض'}
CHARS = "".join(sorted(CHAR_SET_NEW))
PUNCTUATIONS = "".join(sorted([ch for ch in ",.؟!،؛?"]))

# If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
RESTORE_PATH = "/data/asr/workspace/audio/tts/models/ytts/model.pt"

EXPMT_PATH = f"/data/asr/workspace/audio/tts/expmt/ytts/{EXP_ID}"
# This parameter is useful to debug, it skips the training epochs and just do the evaluation and produce the test
# sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE_TRAIN = 48
BATCH_SIZE_EVAL = 16

# Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this
# you might need to re-download the dataset !!) Note: If you add new datasets, please make sure that the dataset
# sampling rate and this parameter are matching, otherwise resample your audios
SAMPLE_RATE = 16000

# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
MAX_AUDIO_LEN_IN_SECONDS = 24

# Download VCTK dataset
IIAI_DATASET_PATH = "/data/asr/workspace/audio/tts"
# Define the number of threads used during the audio resampling
NUM_RESAMPLE_THREADS = 10


# init configs
def get_dataset(index: int = 1):
    return BaseDatasetConfig(
        formatter="iiai_tts",
        dataset_name=f"{index}",
        meta_file_train=f"data/audio/manifest/manifest_se_17072023_32b_{EXP_ID}_{index}_ufid_spk_opt.json",
        meta_file_val=f"data/audio/manifest/manifest_se_17072023_32b_eval_{EXP_ID}_{index}_ufid_spk_opt.json",
        path=IIAI_DATASET_PATH,
        language="ar",
    )


# Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just
# VCTK. Note: If you want to add new datasets, just add them here, and it will automatically compute the speaker
# embeddings (d-vectors) for this new dataset :)
DATASETS_CONFIG_LIST = [get_dataset(ds_index) for ds_index in [1, 2, 3, 4, 5]]

# Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    f"/data/asr/workspace/audio/tts/models/ytts/se/se_model_{EXP_ID}.pt"
)
SPEAKER_ENCODER_CONFIG_PATH = "/data/asr/workspace/audio/tts/models/ytts/se/config.json"

D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

# Iterates all the dataset configs checking if the speakers embeddings are already computed, if not compute it
for dataset_conf in DATASETS_CONFIG_LIST:
    # Check if the embeddings weren't already computed, if not compute it
    embeddings_file = os.path.join(EXPMT_PATH, f"speakers_{EXP_ID}_{dataset_conf.dataset_name}.pth")
    if not os.path.isfile(embeddings_file):
        print(f">>> Computing the speaker embeddings for the ds_{dataset_conf.dataset_name} dataset")
        print(f">>> embeddings_file: {embeddings_file}")
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_spakers_file=None,
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
    mel_fmin=0,
    num_mels=80,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    num_layers_flow=18,
    num_heads_text_encoder=8,
    num_layers_text_encoder=24,
    hidden_channels=256,
    hidden_channels_ffn_text_encoder=1024,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="1",
    use_speaker_encoder_as_loss=True,
    noise_scale=0.9,
    # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the
    # ResNet blocks type 1 like the VITS model Useful parameters to enable the Speaker Consistency Loss (SCL)
    # described in the paper use_speaker_encoder_as_loss=True, Useful parameters to enable multilingual training
    # use_language_embedding=True, embedded_language_dim=4,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    epochs=1000,
    lr_gen=0.00010,
    lr_disc=0.00010,
    # grad_clip=[10.0, 10.0],
    output_path=EXPMT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YTTS-IIAI",
    run_description="""
            - Original YourTTS trained using IIAI dataset
        """,
    dashboard_logger="tensorboard",
    audio=audio_config,
    batch_size=BATCH_SIZE_TRAIN,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE_EVAL,
    num_loader_workers=workers,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=1000,
    save_n_checkpoints=10,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    use_phonemes=False,
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="ar_bw_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        characters=CHARS,
        punctuations=PUNCTUATIONS,
        is_unique=True,
        is_sorted=True,
    ),
    precompute_num_workers=workers,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    test_sentences=[
        [
            "وَبَحْثَ بلينكنْ خِلالَ اجْتِماعِهِ بولي العَهْدِ السُّعوديِّ ، قَضايا تَتَعَلَّقُ بِحُقوقِ الإِنْسانِ ، وَفْقَ ما ذَكَرَ مَسْؤولٌ أَمْريكيٌّ.",
            "spk_57760",
            None,
            "ar",
        ],
        [
            "وَشَمَلتْ قائِمَةُ المُحْتَجَزينَ ، رَجُلَ الدّينِ البارِزِ سَلْمانَ العودَةِ وَأَبْناءِ رَئيسِ المُخابَراتِ السّابِقِ سَعْدُ الجابِريِّ والْمُدافِعِ عَنْ حُقوقِ الإِنْسانِ مُحَمَّدُ القَحْطاني وَعامِلُ الإِغاثَةِ عبدالرحمنْ السُّدحانِ .",
            "spk_58233",
            None,
            "ar",
        ],
        [
            "وَرَجَّحَ كَبيرُ مُسْتَشاري مَرْكَزِ أَبْحاثِ مُؤَسَّسَةِ الدِّفاعِ عَنْ الحُرّيّاتِ في واشِنْطُنْ ، أَنَّ يَكونَ أَهَمُّ عُنْصُرٍ في زيارَةِ بلينكنْ إِلَى الرّياضِ ، هوَ التَّشْجيعُ عَلَى عَدَمِ تَقارُبِ العَلاقاتِ بَيْنَ الصّينِ والسُّعوديَّةِ .",
            "spk_16867",
            None,
            "ar",
        ],
        [
            "وَجاءَ في بَيانٍ صادِرٍ عَنْ وِزارَةِ الخارِجيَّةِ الأَمْريكيَّةِ أَنَّ بلينكنْ وَبِنْ سَلْمانَ ناقَشا سُبُلَ التَّعاوُنِ الِاقْتِصاديِّ وَخاصَّةً في مَجاليَ الطّاقَةِ النَّظيفَةِ والتِّكْنولوجْيا .",
            "spk_39539",
            None,
            "ar",
        ],
        [
            "وَحَثَّتْ السُّلُطاتُ أَرْبابَ العَمَلِ عَلَى عَدَمِ دَفْعِ أَيِّ شَيْءٍ إِذا طَلَبَ المُتَسَلِّلونَ فِدْيَةً .",
            "spk_49442",
            None,
            "ar",
        ],
        [
            "وَجاءَ في بَيانٍ صادِرٍ عَنْ وِزارَةِ الخارِجيَّةِ الأَمْريكيَّةِ أَنَّ بلينكنْ وَبِنْ سَلْمانَ ناقَشا سُبُلَ التَّعاوُنِ الِاقْتِصاديِّ وَخاصَّةً في مَجاليَ الطّاقَةِ النَّظيفَةِ والتِّكْنولوجْيا .",
            "spk_4459",
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
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH, use_accelerate=True),
    config,
    output_path=EXPMT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
