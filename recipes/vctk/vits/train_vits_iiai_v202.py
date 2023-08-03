import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig, CharactersConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

DATA_ID = "google"
RUN_EXP_ID = "v202"
BASE_PATH = "/data/asr/workspace/audio/tts"
EXPMT_PATH = os.path.join(BASE_PATH, f"expmt/vits/{RUN_EXP_ID}")
PHONEME_CACHE = "/data/asr/workspace/audio/tts/mn1/phoneme_cache/v89"
SAMPLE_RATE = 22050
NUM_SPEAKERS = 4

CHARACTERS = "".join(sorted(set([ch for ch in "ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىيًٌٍَُِّْ"])))
PUNCTUATIONS = "".join(sorted(set([ch for ch in "'()-: ,.؟!،؛?"])))

RESTORE_PATH = "/data/asr/workspace/audio/tts/expmt/vits/v91/vits_iiai-July-31-2023_02+59PM-452d4855/best_model_1351298.pth"
MAX_AUDIO_LEN_IN_SECONDS = 25


def get_dataset(index: int = 1):
    return BaseDatasetConfig(
        formatter="iiai_tts",
        dataset_name=f"{index}",
        meta_file_train=f"data/tts/manifest/{DATA_ID}/manifest_{index}_{SAMPLE_RATE}sr.json",
        meta_file_val=f"data/tts/manifest/{DATA_ID}/manifest_{index}_{SAMPLE_RATE}sr_eval.json",
        path=BASE_PATH,
        language="ar",
    )


ds_indexes = [1, 2, 3, 4, 5, 6, 7, 8]
DATASETS_CONFIG_LIST = [get_dataset(ds_index) for ds_index in ds_indexes]

audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    fft_size=1024,
    win_length=1024,
    hop_length=256,
    num_mels=240,
    mel_fmin=0,
)

vits_args = VitsArgs(
    use_speaker_embedding=True,
    use_sdp=False,
    num_layers_flow=32,
    num_layers_posterior_encoder=32,
    dropout_p_text_encoder=0.3,
)

config = VitsConfig(
    model_args=vits_args,
    audio=audio_config,
    run_name="vits_iiai",
    batch_size=48,
    eval_batch_size=4,
    batch_group_size=5,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10000,
    use_phonemes=False,
    # phonemizer="espeak",
    # phoneme_language="ar",
    # phoneme_cache_path=PHONEME_CACHE,
    precompute_num_workers=16,
    max_audio_len=MAX_AUDIO_LEN_IN_SECONDS * SAMPLE_RATE,
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
    text_cleaner="ar_bw_cleaners",
    add_blank=True,
    test_sentences=[
        ["مِن أَجْل تَأْمِين مُوَاطِنِيهَا الْمَوْجُودِين فِي السودَانْ"],
        ["اعْتَاد سُورِيون عَلَى وُجُود كُرْسِي فَارِغ حَوْل حَوْل طَاوِلَة الْإِفْطَار لِلْأَسَفْ"],
        ["بِالتَّعَاوُن مَع الْأَميرِكِي فِي قَرَارْ"],
        ["و مُنْدَفِعُون لِلْعَمَلْ"],
        [
            "يَعْنِي هُنَاك مَسَاعِي مِن عَدْددُول الْحُصُول عَلَى عُضْوِيَّة الْبَرْكَس السُّعُودِيَّة وَاحِدَة مِن هَذِه الدوَل مَا هِي الِاسْتِفَادَة اَلَّتِي تَبْحَث عَنْهَا الريَاضْ"],
        ["الْبُرُوفَيْسُور مُصْطَفَى جرْدْ"],
        ["الطَّرَفَيْن بِأَي مَعْنَى هَلْ"],
        [
            "عَلَى الْمُسْتَوَى الأمْنِي وَالْعَسْكَرِي وَالنَّفْسِي وَالسيَاسِي اَلَّذِي حَاكَتْه أَمْريْكَا وَحُلَفَائِهَا وَلا وَأَنَا لا نَنْسَى تَمَامًا هِي الْكَبنْبَالَات بِمَا يُسَمَّى بِأَصْدِقَاء سُوريا"],
        ["وَمِن الْمَعْلُوم أَن الليثْيُوم يُسْتَخْرَج مِن الْمُسَطَّحَات الْملْحِيَّة"],
        ["وَتَعْمَل التقْنِية عَن طَرِيق تَغْذِيفْ"],
        [
            "وقد لجأ أكبر حزب معارض في جنوب أفريقيا، حزب التحالف الديمقراطي إلى المحكمة لمحاولة إجبار السلطات على اعتقال بوتين إذا ما وطئت قدمه البلاد"],
        [
            "وشهد الشهر الماضي وصول مهمة سلام أفريقية إلى الدول الأوروبية حيث كان الرؤساء الأفارقة يأملون في أن يتمكنوا من جلب أوكرانيا وروسيا إلى طاولة المفاوضات، لكنهم فشلوا في النهاية"],

    ],
    compute_input_seq_cache=True,
    print_step=100,
    save_all_best=True,
    print_eval=True,
    # use_weighted_sampler=True,
    # weighted_sampler_attrs={"speaker_name": 1.0},
    mixed_precision=True,
    output_path=EXPMT_PATH,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    distributed_url="tcp://localhost:63347",
)

# INITIALIZE THE AUDIO PROCESSOR.
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER.
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

# init model
model = Vits(config, ap, tokenizer, speaker_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH),
    config,
    output_path=EXPMT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
