import json
import os

import torch

from TTS.tts.utils.managers import save_file
from TTS.tts.utils.speakers import SpeakerManager

EXP_ID = "v3_ML"
EXPMT_PATH = f"expmt/ytts/{EXP_ID}"
SPK_EMBEDDING_VERSION = "v1"
BASE_PATH = "/data/asr/workspace/audio/tts"
SPK_EMB_CACHE_PATH = os.path.join(EXPMT_PATH, f"spk_emb_{SPK_EMBEDDING_VERSION}")
MAX_SAVE_LIMIT = 200000

os.makedirs(SPK_EMB_CACHE_PATH, exist_ok=True)

SPEAKER_ENCODER_CHECKPOINT_PATH = "expmt/se/multi/v8/run-August-22-2023_09+48AM-452d4855/checkpoint_137000.pth"
SPEAKER_ENCODER_CONFIG_PATH = "expmt/se/multi/v8/run-August-22-2023_09+48AM-452d4855/config.json"

manifests_map = {"hi_org": ["data/audio/multi_lang/manifest_org_hi_ns_phoneme.json",
                            "data/audio/multi_lang/manifest_org_hi_ns_phoneme_eval.json"]}

encoder_manager = SpeakerManager(
    encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    use_cuda=torch.cuda.is_available(),
)

d_vector_files = []
embedding_names = set()
for d_name, manifests in manifests_map.items():
    spk_emb_index = 0
    spk_emb_count = 0
    speaker_mapping = {}
    count = 0
    cur_manifest = None
    d_cache_dir = os.path.join(SPK_EMB_CACHE_PATH, f"{d_name}")
    os.makedirs(d_cache_dir, exist_ok=True)
    for m_index, manifest in enumerate(manifests):
        cur_manifest = manifest
        with open(manifest, encoding="utf-8") as sm:
            for line in sm:
                jd = json.loads(line.strip("\n").strip())
                speaker = jd["speaker"]
                audio_filepath = jd["audio_filepath"]
                u_fid = jd['u_fid']
                audio_unique_name = f"{d_name}#{u_fid}"
                if audio_unique_name in embedding_names:
                    continue
                spk_emb = encoder_manager.compute_embedding_from_clip(audio_filepath)
                speaker_mapping[audio_unique_name] = {"name": speaker, "embedding": spk_emb}
                embedding_names.add(audio_unique_name)
                if spk_emb_count >= MAX_SAVE_LIMIT:
                    mapping_file_path = os.path.join(d_cache_dir, f"spk_emb_{d_name}_{spk_emb_index}.pth")
                    save_file(speaker_mapping, mapping_file_path)
                    d_vector_files.append(mapping_file_path)
                    print(
                        f"Speaker embeddings saved at: {mapping_file_path}, count: {count}, manifest: {cur_manifest}, embeddings: {len(embedding_names)}")
                    spk_emb_index = spk_emb_index + 1
                    spk_emb_count = 0
                    speaker_mapping = {}
                else:
                    spk_emb_count = spk_emb_count + 1

                count = count + 1
                if count % 1000 == 0:
                    print(
                        f"Done: {count}, d_name: {d_name}, manifest: {cur_manifest}, embeddings: {len(embedding_names)}")

    if len(speaker_mapping) > 0:
        mapping_file_path = os.path.join(d_cache_dir, f"spk_emb_{d_name}_{spk_emb_index}.pth")
        save_file(speaker_mapping, mapping_file_path)
        print(
            f"Speaker embeddings saved at: {mapping_file_path}, count: {count}, manifest: {cur_manifest}, embeddings: {len(embedding_names)}")
        d_vector_files.append(mapping_file_path)

    d_vector_cache_dir = os.path.join(BASE_PATH, d_cache_dir)
    d_vector_cache_path = os.path.join(d_vector_cache_dir, "d_vector_files.json")

    with open(d_vector_cache_path, encoding="utf-8", mode="w") as dvcp:
        json.dump(d_vector_files, dvcp)
