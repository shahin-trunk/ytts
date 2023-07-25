from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig, CharactersConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

EXP_ID = "v73"
BASE_PATH = "/data/asr/workspace/audio/tts"
EXPMT_PATH = f"{BASE_PATH}/expmt/vits/{EXP_ID}"
SAMPLE_RATE = 22050

CHAR_SET_NEW = {'ه', 'إ', 'ي', 'ق', 'ّ', 'س', 'ع', 'ئ', 'ن', 'ث', 'ف', 'ة', 'ت', 'ً', 'ٍ', 'و', 'ب', 'ص', 'ى', 'د', 'غ',
                'ذ', 'ك', 'ج', 'َ', 'م', 'خ', ' ', 'ٌ', 'ء', 'ش', 'ؤ', 'ظ', 'ر', 'ل', 'ز', 'ُ', 'آ', 'ِ', 'ا', 'ح', 'ْ',
                'ط', 'أ', 'ض'}
CHARS = "".join(sorted(CHAR_SET_NEW))
PUNCTUATIONS = "".join(sorted([ch for ch in ",.؟!،؛?"]))

RESTORE_PATH = "/data/asr/workspace/audio/tts/models/vits/ar/model_16f.pt"


def get_dataset(index: int = 1):
    return BaseDatasetConfig(
        formatter="iiai_tts",
        dataset_name=f"{index}",
        meta_file_train=f"data/audio/manifest/manifest_32b_{EXP_ID}_{index}_ufid_spk_opt_sr{SAMPLE_RATE}.json",
        meta_file_val=f"data/audio/manifest/manifest_32b_eval_{EXP_ID}_{index}_ufid_spk_opt_sr{SAMPLE_RATE}.json",
        path=BASE_PATH,
        language="ar",
    )


DATASETS_CONFIG_LIST = [get_dataset(ds_index) for ds_index in [1, 2, 3, 4, 5]]

audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
)

vits_args = VitsArgs(
    use_speaker_embedding=True,
    num_layers_flow=18,
    num_heads_text_encoder=8,
    num_layers_text_encoder=24,
    hidden_channels=256,
    hidden_channels_ffn_text_encoder=1024,
)

config = VitsConfig(
    model_args=vits_args,
    audio=audio_config,
    run_name="vits_iiai",
    batch_size=16,
    eval_batch_size=8,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10000,
    text_cleaner="ar_bw_cleaners",
    add_blank=True,
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
    use_phonemes=False,
    compute_input_seq_cache=True,
    print_step=100,
    print_eval=False,
    mixed_precision=True,
    output_path=EXPMT_PATH,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
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
