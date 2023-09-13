import os

from TTS.encoder.configs.speaker_encoder_config import SpeakerEncoderConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig

# Definitions ###
BASE_PATH = "/data/asr/workspace/audio/tts"
DATA_PATH = "data/audio/multi_lang"
VERSION = "v9"
SAMPLE_RATE = 22050

# dataset
# download: https://www.openslr.org/17/
RIR_SIMULATED_PATH = os.path.join(BASE_PATH, "models/speaker_encoding/RIRS_NOISES/simulated_rirs")

# download: https://www.openslr.org/17/
MUSAN_PATH = os.path.join(BASE_PATH, "models/speaker_encoding/musan/")

LANGUAGE = "multi"

# training
OUTPUT_PATH = os.path.join(BASE_PATH, f"expmt/se/{LANGUAGE}/{VERSION}")
os.makedirs(OUTPUT_PATH, exist_ok=True)
CONFIG_OUT_PATH = os.path.join(OUTPUT_PATH, "config_se.json")
RESTORE_PATH = os.path.join(BASE_PATH, "expmt/se/multi/v9/run-August-30-2023_04+16PM-452d4855/checkpoint_12000.pth")
# RESTORE_PATH = None

# instance the config
# to speaker encoder
config = SpeakerEncoderConfig()


# to emotion encoder
# config = EmotionEncoderConfig()


def get_dataset(manifest_train: str, d_name: str, lang: str = "ar", ignored_speakers=None):
    if ignored_speakers is None:
        ignored_speakers = []
    return BaseDatasetConfig(
        formatter="iiai_se",
        dataset_name=f"{lang}_{d_name}",
        meta_file_train=os.path.join(DATA_PATH, manifest_train),
        path=BASE_PATH,
        language=lang,
        ignored_speakers=ignored_speakers,
    )


DATASETS_CONFIG_LIST = [
    get_dataset(manifest_train="manifest_gen_ar_ns.json",
                d_name="gen",
                lang="ar"),
    get_dataset(manifest_train="manifest_org_ar_ns.json",
                d_name="org",
                lang="ar"),
    # get_dataset(manifest_train="manifest_org_en_ns_clean.json",
    #             d_name="org",
    #             lang="en"),
    get_dataset(manifest_train="manifest_gen_en_ns.json",
                d_name="gen",
                lang="en"),
    get_dataset(manifest_train="manifest_org_hi_ns.json",
                d_name="org",
                lang="hi")
]

# add the dataset to the config
config.datasets = DATASETS_CONFIG_LIST

# ### TRAINING CONFIG ####
# The encoder data loader balancer the dataset item equally to guarantee better training and
# to attend the losses requirements. It has two parameters to control the final batch size the number total of
# speaker used in each batch and the number of samples for each speaker

# number total of speaker in batch in training
config.num_classes_in_batch = 64
# number of utterance per class/speaker in the batch in training
config.num_utter_per_class = 4
# final batch size = config.num_classes_in_batch * config.num_utter_per_class

# filter_small_samples below 2.0 sec
config.filter_small_samples = False  # Taking a lot of time if True.

# number total of speaker in batch in evaluation
config.eval_num_classes_in_batch = 32
# number of utterance per class/speaker in the batch in evaluation
config.eval_num_utter_per_class = 2

# number of data loader workers
config.num_loader_workers = 0
config.num_val_loader_workers = 0

# number of epochs
config.epochs = 73000
# loss to be used in training
config.loss = "softmaxproto"

# run eval
config.run_eval = False

# output path for the checkpoints
config.output_path = OUTPUT_PATH

# Save local checkpoint every save_step steps
config.save_step = 3000

# ### Model Config ###
config.model_params = {
    "model_name": "resnet",  # supported "lstm" and "resnet"
    "input_dim": 320,
    "use_torch_spec": True,
    "log_input": True,
    "proj_dim": 2560,  # embedding dim
}

# Audio Config ### To fast train the model divides the audio in small parts. This parameter defines the length in
# seconds of these "parts"
config.voice_len = 1.3

# all others configs
config.audio = {
    "fft_size": 2048,
    "win_length": 2048,
    "hop_length": 256,
    "frame_shift_ms": None,
    "frame_length_ms": None,
    "stft_pad_mode": "reflect",
    "sample_rate": SAMPLE_RATE,
    "resample": False,
    "preemphasis": 0.97,
    "ref_level_db": 20,
    "do_sound_norm": False,
    "do_trim_silence": False,
    "trim_db": 60,
    "power": 1.5,
    "griffin_lim_iters": 60,
    "num_mels": 320,
    "mel_fmin": 0.0,
    "mel_fmax": 8000.0,
    "spec_gain": 20,
    "signal_norm": False,
    "min_level_db": -100,
    "symmetric_norm": False,
    "max_norm": 4.0,
    "clip_norm": False,
    "stats_path": None,
    "do_rms_norm": True,
    "db_level": -27.0,
}

# config.audio = {
#     "sample_rate": SAMPLE_RATE,
#     "fft_size": 2048,
#     "win_length": 1024,
#     "hop_length": 256,
#     "num_mels": 320,
#     "do_trim_silence": False,
#     "mel_fmax": 11024,
# }

# Augmentation Config ###
# config.audio_augmentation = {
#     # additive noise and room impulse response (RIR) simulation similar to: https://arxiv.org/pdf/2009.14153.pdf
#     "p": 0.00,  # probability to the use of one of the augmentation - 0 means disabled
#     "rir": {"rir_path": RIR_SIMULATED_PATH, "conv_mode": "full"},  # download: https://www.openslr.org/17/
#     "additive": {
#         "sounds_path": MUSAN_PATH,
#         "speech": {"min_snr_in_db": 13, "max_snr_in_db": 20, "min_num_noises": 1, "max_num_noises": 1},
#         "noise": {"min_snr_in_db": 0, "max_snr_in_db": 15, "min_num_noises": 1, "max_num_noises": 1},
#         "music": {"min_snr_in_db": 5, "max_snr_in_db": 15, "min_num_noises": 1, "max_num_noises": 1},
#     },
#     "gaussian": {"p": 0.7, "min_amplitude": 0.0, "max_amplitude": 1e-05},
# }

config.save_json(CONFIG_OUT_PATH)

print(CONFIG_OUT_PATH)
if RESTORE_PATH is not None:
    command = f"python3 {BASE_PATH}/TTS/TTS/bin/train_encoder.py --config_path {CONFIG_OUT_PATH} --restore_path {RESTORE_PATH}"
else:
    command = f"python3 {BASE_PATH}/TTS/TTS/bin/train_encoder.py --config_path {CONFIG_OUT_PATH}"

os.system(command)
