import logging
import os
from uuid import uuid4

import torch

from TTS.tts.utils.managers import save_file
from TTS.tts.utils.speakers import SpeakerManager

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("MSE")

BASE_PATH = "/data/asr/workspace/audio/tts"

tgt_emb_path = os.path.join(BASE_PATH, "expmt/ytts/v35_ML/spk_emb_v1/spk_emb_ext_v3.pth")

SPEAKER_ENCODER_CHECKPOINT_PATH = os.path.join(BASE_PATH,
                                               "expmt/se/multi/v9/run-August-30-2023_09+59PM-452d4855/checkpoint_69000.pth")
SPEAKER_ENCODER_CONFIG_PATH = os.path.join(BASE_PATH,
                                           "expmt/se/multi/v9/run-August-30-2023_09+59PM-452d4855/config.json")

src_wavs = {
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/vp/ar_spk_vp_01_1.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/vp/ar_spk_vp_01_2.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/vp/ar_spk_vp_01_3.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/vp/ar_spk_vp_01_4.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/vp/ar_spk_vp_01_5.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/vp/ar_spk_vp_01_6.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/vp/ar_spk_vp_01_7.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/vp/ar_spk_vp_01_8.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/vp/ar_spk_vp_02_1.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/vp/ar_spk_vp_02_2.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/vp/ar_spk_vp_03_1.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_01.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_02.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_03.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_04.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_05.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_06.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_07.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_08.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_09.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_10.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_11.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_12.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_13.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_14.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_15.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_16.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_17.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_18.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_19.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_20.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_21.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_22.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_23.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_24.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_25.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_26.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_27.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_28.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_29.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_30.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_31.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_32.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_33.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_34.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_35.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_36.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_37.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_38.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_39.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_40.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_41.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_42.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_43.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/ar/el/ar_spk_el_44.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_01.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_02.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_03.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_04.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_05.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_06.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_07.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_08.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_09.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_10.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_11.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_12.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_13.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_14.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/el/en_spk_el_15.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_01.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_02_1.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_02_2.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_02_3.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_02_4.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_02_5.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_02_6.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_03_1.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_03_2.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_03_3.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_03_4.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_03_5.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_04_1.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_04_2.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_04_3.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_04_4.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_04_5.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_04_6.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_04_7.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_05_1.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_05_2.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_05_3.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_05_4.wav",
    "/data/asr/workspace/audio/tts/data/audio/wav/en/vp/en_spk_vp_05_5.wav",
}

encoder_manager = SpeakerManager(
    encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    use_cuda=torch.cuda.is_available(),
)

speaker_emb_mapping = {}
for fp in src_wavs:
    spk_name = os.path.basename(fp).replace(".wav", "")
    spk_emb = encoder_manager.compute_embedding_from_clip(fp)
    speaker_emb_mapping[str(uuid4())] = {"name": spk_name, "embedding": spk_emb}
    _log.info(f"Added embedding, clip: {fp}, speaker: {spk_name}, emb_dim: {len(spk_emb)}")

save_file(speaker_emb_mapping, tgt_emb_path)
_log.info(f"Saved final d_vector_file at: {tgt_emb_path}, size: {len(speaker_emb_mapping)}")
