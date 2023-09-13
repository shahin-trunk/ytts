from TTS.tts.utils.managers import load_file

src_emb_path = "expmt/ytts/v34_ML/spk_emb_v1/ar_cmb/spk_emb_ar_cmb_3.pth"
spk_mapping = load_file(src_emb_path)
print(list(spk_mapping[list(spk_mapping.keys())[1200]]["embedding"])[:10])


src_emb_path = "expmt/ytts/v33_ML/spk_emb_v1/ar_cmb/spk_emb_ar_cmb_3.pth"
spk_mapping = load_file(src_emb_path)
print(list(spk_mapping[list(spk_mapping.keys())[1200]]["embedding"])[:10])
