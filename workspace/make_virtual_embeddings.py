import json
import logging
import os
import random
from itertools import combinations
from uuid import uuid4

import numpy as np
import torch

from TTS.tts.utils.managers import save_file, load_file
from TTS.tts.utils.speakers import SpeakerManager

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("MSE")

MAX_NUM_VS_SPK = 64
NUM_VS_SPK_TO_MERGE = 3
BASE_PATH = "/data/asr/workspace/audio/tts"

tgt_emb_path = os.path.join(BASE_PATH, "expmt/ytts/v35_ML/spk_emb_v1/spk_emb_cmb_inf_virtual_{}.pth")

SPEAKER_ENCODER_CHECKPOINT_PATH = os.path.join(BASE_PATH,
                                               "expmt/se/multi/v9/run-August-30-2023_09+59PM-452d4855/checkpoint_69000.pth")
SPEAKER_ENCODER_CONFIG_PATH = os.path.join(BASE_PATH,
                                           "expmt/se/multi/v9/run-August-30-2023_09+59PM-452d4855/config.json")
D_VECTOR_FILES_CACHE_MAP = {
    "ar": os.path.join(BASE_PATH, "expmt/ytts/v35_ML/spk_emb_v1/ar_cmb/d_vector_files.json"),
    "en": os.path.join(BASE_PATH, "expmt/ytts/v35_ML/spk_emb_v1/en_cmb/d_vector_files.json"),
}

speaker_emb_mapping = {}
for lang, dv_cache in D_VECTOR_FILES_CACHE_MAP.items():
    speaker_emb_mapping_lang = {}
    speaker_u_sid_mapping_lang = {}
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

            _log.debug(f"name_to_id: {encoder_manager.name_to_id}")
            _log.info(f"speaker_names: {encoder_manager.speaker_names}")

            speaker_names = list(encoder_manager.speaker_names)

            for spk_name in speaker_names:
                cur_emb = None
                u_sid = None
                if spk_name in speaker_u_sid_mapping_lang:
                    cur_emb = speaker_emb_mapping_lang[speaker_u_sid_mapping_lang[spk_name]]["embedding"]
                    u_sid = speaker_u_sid_mapping_lang[spk_name]
                    _log.info(f"Embedding exists for speaker_name: {spk_name}, u_sid: {u_sid}")

                spk_emb = encoder_manager.get_mean_embedding(spk_name)
                if cur_emb is not None:
                    spk_emb = np.stack([cur_emb, spk_emb]).mean(0)
                    _log.info(
                        f"Combined embedding, speaker_name: {spk_name}, emb_dim_cur: {cur_emb.shape}, emb_dim_new: {spk_emb.shape}")

                if u_sid is None:
                    u_sid = str(uuid4())

                speaker_emb_mapping_lang[u_sid] = {"name": spk_name, "embedding": spk_emb}
                speaker_u_sid_mapping_lang[spk_name] = u_sid
                _log.info(
                    f"Added mean embedding, speaker: {spk_name}, u_sid: {u_sid}, emb_dim: {spk_emb.shape}, lang: {lang}, virtual:{False}\n")

            spk_merge_combs = list(combinations(speaker_names, NUM_VS_SPK_TO_MERGE))

            _log.info(f"spk_merge_combs_total: {len(spk_merge_combs)}")
            if len(spk_merge_combs) > MAX_NUM_VS_SPK:
                spk_merge_combs = random.choices(spk_merge_combs, k=MAX_NUM_VS_SPK)

            _log.info(f"spk_merge_combs_sel: {len(spk_merge_combs)}")

            for spk_merge_comb in spk_merge_combs:
                sel_spk_names = list(spk_merge_comb)
                spk_name = "|".join(sel_spk_names)
                cur_emb = None
                u_sid = None
                if spk_name in speaker_u_sid_mapping_lang:
                    cur_emb = speaker_emb_mapping_lang[speaker_u_sid_mapping_lang[spk_name]]["embedding"]
                    u_sid = speaker_u_sid_mapping_lang[spk_name]
                    _log.info(f"Embedding exists for speaker_name: {spk_name}, u_sid: {u_sid}")

                sel_spk_embeds = [encoder_manager.get_mean_embedding(spk_name) for spk_name in sel_spk_names]
                spk_emb = np.stack(sel_spk_embeds).mean(0)
                if cur_emb is not None:
                    spk_emb = np.stack([cur_emb, spk_emb]).mean(0)
                    _log.info(
                        f"Combined embedding speaker_name: {spk_name}, emb_dim_cur: {cur_emb.shape}, emb_dim_new: {spk_emb.shape}")

                if u_sid is None:
                    u_sid = str(uuid4())

                speaker_emb_mapping_lang[u_sid] = {"name": spk_name, "embedding": spk_emb}
                speaker_u_sid_mapping_lang[spk_name] = u_sid

                _log.info(
                    f"Added mean embedding, speaker: {spk_name}, u_sid: {u_sid}, emb_dim: {spk_emb.shape}, lang: {lang}, virtual:{True}\n")

    tgt_emb_path_lang = tgt_emb_path.format(lang)
    save_file(speaker_emb_mapping_lang, tgt_emb_path_lang)
    _log.info(f"Saved d_vector_file at: {tgt_emb_path_lang}, size: {len(speaker_emb_mapping_lang)}")
    speaker_emb_mapping.update(speaker_emb_mapping_lang)

tgt_emb_path_all = tgt_emb_path.format("all_lang")
save_file(speaker_emb_mapping, tgt_emb_path_all)
_log.info(f"Saved final d_vector_file at: {tgt_emb_path_all}, size: {len(speaker_emb_mapping)}")


## FOR adding additional speakers
# tgt_emb_path_all = tgt_emb_path.format("all_lang_v2")
# speaker_emb_mapping = load_file(tgt_emb_path_all)
# _log.info(f"Loaded: {tgt_emb_path_all}, size: {len(speaker_emb_mapping)}")
#
# encoder_manager = SpeakerManager(
#     encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
#     encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
#     use_cuda=torch.cuda.is_available(),
# )
#
# emb_spk_arb_male_8 = encoder_manager.compute_embedding_from_clip("/data/asr/workspace/audio/tts/data/arabic/male8/wav/22k_denoise/otnYfWDcYDA-9.wav")
# emb_spk_arb_female_2 = encoder_manager.compute_embedding_from_clip("/data/asr/workspace/audio/tts/data/arabic/female2/wav/22k_denoise/P2GCEgN3wxs-54.wav")
#
# speaker_emb_mapping[str(uuid4())] = {"name": "spk_arb_male_8", "embedding": emb_spk_arb_male_8}
# speaker_emb_mapping[str(uuid4())] = {"name": "spk_arb_female_2", "embedding": emb_spk_arb_female_2}
#
# tgt_emb_path_all = tgt_emb_path.format("all_lang_v3")
# save_file(speaker_emb_mapping, tgt_emb_path_all)
# _log.info(f"Saved final d_vector_file at: {tgt_emb_path_all}, size: {len(speaker_emb_mapping)}")
