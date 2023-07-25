import os

from TTS.bin.compute_embeddings import compute_embeddings

run_index = 3
EXP_ID = "v73"
EXPMT_PATH = f"/data/asr/workspace/audio/tts/expmt/ytts/{EXP_ID}"
os.makedirs(EXPMT_PATH, exist_ok=True)
# Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    f"/data/asr/workspace/audio/tts/models/ytts/se/se_model_{EXP_ID}.pt"
)
SPEAKER_ENCODER_CONFIG_PATH = "/data/asr/workspace/audio/tts/models/ytts/se/config.json"
IIAI_DATASET_PATH = "/data/asr/workspace/audio/tts"

embeddings_file = os.path.join(EXPMT_PATH, f"speakers_{EXP_ID}_{run_index}.pth")
if not os.path.isfile(embeddings_file):
    compute_embeddings(
        SPEAKER_ENCODER_CHECKPOINT_PATH,
        SPEAKER_ENCODER_CONFIG_PATH,
        embeddings_file,
        old_spakers_file=None,
        config_dataset_path=None,
        formatter_name="iiai_tts",
        dataset_name=f"{run_index}",
        dataset_path=IIAI_DATASET_PATH,
        meta_file_train=f"data/audio/manifest/manifest_se_17072023_32b_{EXP_ID}_{run_index}_ufid_spk_opt.json",
        meta_file_val=f"data/audio/manifest/manifest_se_17072023_32b_eval_{EXP_ID}_{run_index}_ufid_spk_opt.json",
        disable_cuda=False,
        no_eval=False,
    )
