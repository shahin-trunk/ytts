import json
import logging

import torch
# from TTS.tts.utils.speakers import SpeakerManager
from sklearn.cluster import MeanShift, MiniBatchKMeans

from TTS.tts.utils.managers import load_file

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("MSE")

src = "data/audio/multi_lang/sel/manifest_org_en_kf_raw_emb.json"
tgt = "data/audio/multi_lang/sel/manifest_org_en_kf_raw_emb_spk.json"

SPEAKER_ENCODER_CHECKPOINT_PATH = "expmt/se/multi/v9/run-August-30-2023_09+59PM-452d4855/checkpoint_66000.pth"
SPEAKER_ENCODER_CONFIG_PATH = "expmt/se/multi/v9/run-August-30-2023_09+59PM-452d4855/config.json"

# encoder_manager = SpeakerManager(
#     encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
#     encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
#     use_cuda=torch.cuda.is_available(),
# )

spk_emb_map = {}

with open(src, encoding="utf-8") as sm:
    for index, line in enumerate(sm):
        jd = json.loads(line.strip("\n").strip())
        audio_filepath = jd["audio_filepath"]
        emb_filepath = jd["emb_filepath"]
        u_fid = jd["u_fid"]
        spk_emb_map[u_fid] = load_file(emb_filepath)
        if index % 100 == 0:
            _log.info(f"Done loading: {index}")

emb_data = torch.stack([torch.Tensor(spk_emb) for spk_emb in spk_emb_map.values()])
_log.info(f"Data: {emb_data.shape}")
# model = MeanShift()
model = MiniBatchKMeans(n_clusters=120, batch_size=1024, verbose=10, max_iter=700)
spk_pred = model.fit_predict(emb_data)

spk_u_fid_map = {}
for index, u_fid in enumerate(spk_emb_map.keys()):
    spk_u_fid_map[u_fid] = f"spk_{spk_pred[index]}"

with open(src, encoding="utf-8") as sm:
    with open(tgt, encoding="utf-8", mode="w") as tm:
        for index, line in enumerate(sm):
            jd = json.loads(line.strip("\n").strip())
            u_fid = jd["u_fid"]
            jd["speaker"] = spk_u_fid_map[u_fid]

            json.dump(jd, tm)
            tm.write("\n")
            if index % 100 == 0:
                _log.info(f"Done: {index}")

# import json
# import logging
# import os
# import shutil
#
# import torchaudio
#
# # from TTS.tts.utils.speakers import SpeakerManager
#
# logging.basicConfig(level=logging.INFO)
# _log = logging.getLogger("MSE")
#
# src = "data/audio/multi_lang/sel/manifest_org_en_kf_raw_emb_spk.json"
# tgt = "data/audio/multi_lang/sel/manifest_org_en_kf.json"
#
# spk_file_map = {}
#
# with open(src, encoding="utf-8") as sm:
#     with open(tgt, encoding="utf-8", mode="w") as tm:
#         for index, line in enumerate(sm):
#             jd = json.loads(line.strip("\n").strip())
#             spk = f'en_kf_{jd["speaker"]}'
#             # _log.info(spk)
#             src_af = jd["audio_filepath"]
#             spk_dir = os.path.join("data/audio/wav/en/kf", jd["speaker"])
#             os.makedirs(spk_dir, exist_ok=True)
#
#             if spk in spk_file_map:
#                 frequency = spk_file_map[spk]["frequency"] + 1
#                 tgt_af = os.path.join(spk_dir, f"audio_{frequency}.wav")
#                 shutil.copy(src_af, tgt_af)
#                 _log.info(f"{src_af} --> {tgt_af}")
#                 spk_file_map[spk]["audio_files"].append(tgt_af)
#                 spk_file_map[spk]["frequency"] = frequency
#             else:
#                 frequency = 1
#                 tgt_af = os.path.join(spk_dir, f"audio_{frequency}.wav")
#                 shutil.copy(src_af, tgt_af)
#                 _log.info(f"{src_af} --> {tgt_af}")
#                 spk_file_map[spk] = {"audio_files": [tgt_af], "frequency": 1}
#
#             audio_meta_rf = torchaudio.info(tgt_af)
#             duration = float(audio_meta_rf.num_frames / audio_meta_rf.sample_rate)
#
#             metadata = {
#                 "audio_filepath": tgt_af,
#                 "text": jd["text"],
#                 "speaker": spk,
#                 "duration": duration,
#             }
#
#             json.dump(metadata, tm)
#             tm.write("\n")
#
# _log.info(spk_file_map["en_kf_spk_89"])
