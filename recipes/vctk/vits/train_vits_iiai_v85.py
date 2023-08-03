import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig, CharactersConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

torch.set_num_threads(24)

DATA_ID = "v80"
RUN_EXP_ID = "v85"
BASE_PATH = "/data/asr/workspace/audio/tts"
EXPMT_PATH = os.path.join(BASE_PATH, f"expmt/vits/{RUN_EXP_ID}")
PHONEME_CACHE = os.path.join(BASE_PATH, f"expmt/vits/phoneme_cache_v6")
SAMPLE_RATE = 44100
NUM_SPEAKERS = 16
MAX_AUDIO_LEN_IN_SECONDS = 10

CHARACTERS = "".join(sorted(set([ch for ch in "abdfhijklmnqrstuwz|ðħɡɣɹʃʒʔʕˈˌːˤ̪θχ"])))
PUNCTUATIONS = "".join(sorted(set([ch for ch in "'()-: ,.؟!،؛?"])))

RESTORE_PATH = "/data/asr/workspace/audio/tts/models/vits/ar/model_v79best.pt"


def get_dataset(index: int = 1):
    return BaseDatasetConfig(
        formatter="iiai_tts",
        dataset_name=f"{index}",
        meta_file_train=f"data/audio/manifest/{DATA_ID}/manifest_{index}_32b_{NUM_SPEAKERS}spk_{SAMPLE_RATE}sr.json",
        meta_file_val=f"data/audio/manifest/{DATA_ID}/manifest_{index}_32b_{NUM_SPEAKERS}spk_{SAMPLE_RATE}sr_eval.json",
        path=BASE_PATH,
        language="ar",
    )


# DATASETS_CONFIG_LIST = [get_dataset(ds_index) for ds_index in [1, 2, 3, 4, 5]]
DATASETS_CONFIG_LIST = [get_dataset(ds_index) for ds_index in [1]]

audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    win_length=2048,
    fft_size=2048,
    hop_length=512,
    num_mels=240,
    mel_fmin=0,
)

vits_args = VitsArgs(
    use_sdp=True,
    num_layers_flow=32,
    num_layers_dp_flow=16,
    num_layers_posterior_encoder=32,
    num_heads_text_encoder=4,
    num_layers_text_encoder=12,
    out_channels=1025,
    use_speaker_embedding=True,
    upsample_rates_decoder=[8, 8, 2, 2, 2],
    upsample_kernel_sizes_decoder=[16, 16, 4, 4, 4],
    reinit_text_encoder=True,
    reinit_DP=True,
    dropout_p_duration_predictor=0.1,
)

config = VitsConfig(
    lr_gen=0.0014,
    lr_disc=0.0014,
    model_args=vits_args,
    audio=audio_config,
    run_name="vits_iiai",
    batch_size=32,
    eval_batch_size=4,
    batch_group_size=5,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10000,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    use_phonemes=True,
    phonemizer="espeak",
    phoneme_language="ar",
    phoneme_cache_path=PHONEME_CACHE,
    precompute_num_workers=8,
    characters=CharactersConfig(
        characters_class="TTS.tts.utils.text.characters.IPAPhonemes",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=CHARACTERS,
        punctuations=PUNCTUATIONS,
        is_unique=True,
        is_sorted=True,
    ),
    text_cleaner="phoneme_cleaners",
    add_blank=True,
    test_sentences=[
        ["مطالبات بحماية الأطفال من صور الذكاء الاصطناعي المنطوية على انتهاكات جنسية"],
        ["استمرار حرائق الغابات في اليونان مع ارتفاع درجات الحرارة"],
        [
            "تشهد مساحات شاسعة من جنوب أوروبا ارتفاعا متزايدا في درجات حرارة قياسية مع اندلاع حرائق الغابات في جميع أنحاء القارة"],
        [
            "لكن درجات الحرارة المنخفضة خلال الليل ومستويات الرطوبة المرتفعة في الهواء ساعدت رجال الإطفاء في مكافحة الحريق والسيطرة عليه"],
        [
            "وعلى الرغم من العدد المروع الذي يسجل للمهاجرين المفقودين في البحر الأبيض المتوسط إلا أن الرقم الحقيقي أكبر بكثير، بسبب نقص المعلومات وعدم وجود آليات للإبلاغ الرسمي والمنهجي عن حالات الوفاة والاختفاء"],
        [
            "خرج مئات الفلسطينيين في مخيم جنين إلى الشوارع الإثنين احتجاجا على الاعتقالات التي تقوم بها قوات الأمن التابعة للسلطة الفلسطينية"],
        [
            "ووجه رئيس جنوب أفريقيا هذا التحذير قبل أسابيع من عقد اجتماع دولي في العاصمة جوهانسبرغ، من المقرر أن يحضره الرئيس الروسي"],
        [
            "وإذا ما غادر بوتين الأراضي الروسية، فإنه سيكون تحت طائلة مذكرة التوقيف الصادرة من المحكمة الجنائية الدولية بحقه"],
        [
            "وجنوب أفريقيا إحدى الدول الموقعة على معاهدة المحكمة الجنائية الدولية وبالتالي ينبغي أن تساعد في اعتقال بوتين"],
        [
            "وقد لجأ أكبر حزب معارض في جنوب أفريقيا، حزب التحالف الديمقراطي إلى المحكمة لمحاولة إجبار السلطات على اعتقال بوتين إذا ما وطئت قدمه البلاد"],
        [
            "وشهد الشهر الماضي وصول مهمة سلام أفريقية إلى الدول الأوروبية حيث كان الرؤساء الأفارقة يأملون في أن يتمكنوا من جلب أوكرانيا وروسيا إلى طاولة المفاوضات، لكنهم فشلوا في النهاية"],
        [
            "وقد ذكر مرارا عن إحجام الدول الإفريقية عن دعم قرارات الجمعية العامة للأمم المتحدة التي تدين الحرب الروسية في أوكرانيا"],
        [
            "جاءت الاعتقالات بعد أيام فقط من قيام الرئيس الفلسطيني محمود عباس بزيارة إلى مخيم جنين في أعقاب العملية الإسرائيلية هناك كما أشارت تقارير إعلامية إسرائيلية إلى أن إسرائيل ستتراجع عن العمليات في جنين لمنح عباس سلطة استعادة السيطرة على المنطقة"],
    ],
    compute_input_seq_cache=False,
    print_step=10,
    save_all_best=True,
    print_eval=True,
    # use_weighted_sampler=True,
    # weighted_sampler_attrs={"speaker_name": 1.0},
    mixed_precision=False,
    output_path=EXPMT_PATH,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    distributed_url="tcp://localhost:63337",
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
