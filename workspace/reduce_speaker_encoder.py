import json
import logging
import os
from uuid import uuid4

import torch

from TTS.tts.utils.managers import save_file
from TTS.tts.utils.speakers import SpeakerManager

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("MSE")

BASE_PATH = "/data/asr/workspace/audio/tts"

D_VECTOR_FILES_CACHE = [
    os.path.join(BASE_PATH, "expmt/ytts/v35_ML/spk_emb_v1/ar_cmb/d_vector_files.json"),
    os.path.join(BASE_PATH, "expmt/ytts/v35_ML/spk_emb_v1/en_cmb/d_vector_files.json"),
]

tgt_emb_path = os.path.join(BASE_PATH, "expmt/ytts/v35_ML/spk_emb_v1/spk_emb_cmb_inf.pth")

SPEAKER_ENCODER_CHECKPOINT_PATH = os.path.join(BASE_PATH,
                                               "expmt/se/multi/v9/run-August-30-2023_09+59PM-452d4855/checkpoint_69000.pth")
SPEAKER_ENCODER_CONFIG_PATH = os.path.join(BASE_PATH,
                                           "expmt/se/multi/v9/run-August-30-2023_09+59PM-452d4855/config.json")

speaker_emb_mapping = {}
for dv_cache in D_VECTOR_FILES_CACHE:
    with open(dv_cache, encoding="utf-8") as dvc:
        dv_files = list(json.load(dvc))
        _log.info(f"dv_files: {dv_files}")
        for dv_file in dv_files:
            dv_file = dv_file if BASE_PATH in dv_file else os.path.join(BASE_PATH, dv_file)
            encoder_manager = SpeakerManager(
                encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
                encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
                use_cuda=torch.cuda.is_available(),
            )
            _log.info(f"loading dv_file: {dv_file}")
            encoder_manager.load_embeddings_from_file(dv_file)

            _log.info(f"name_to_id: {encoder_manager.name_to_id}")
            _log.info(f"speaker_names: {encoder_manager.speaker_names}")
            for spk_name in encoder_manager.speaker_names:
                u_sid = str(uuid4())
                spk_emb = encoder_manager.get_mean_embedding(spk_name)
                speaker_emb_mapping[u_sid] = {"name": spk_name, "embedding": spk_emb}
                _log.info(f"Added mean embedding, speaker: {spk_name}, u_sid: {u_sid}, emb_dim: {spk_emb.shape}")

save_file(speaker_emb_mapping, tgt_emb_path)
_log.info(f"Saved final reduced d_vector_file at: {tgt_emb_path}, size: {len(speaker_emb_mapping)}")
