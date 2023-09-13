import json

num_utt_spk = 2100
src = "data/audio/multi_lang/sel/manifest_gen_en_ns.json"
tgt = f"data/audio/multi_lang/sel/manifest_gen_en_ns_{num_utt_spk}.json"

spk_utt_map = {}
with open(tgt, encoding="utf-8", mode="w") as tm:
    with open(src, encoding="utf-8") as sm:
        for line in sm:
            jd = json.loads(line.strip("\n").strip())
            speaker = jd["speaker"]
            duration = float(jd["duration"])

            if not (2.0 <= duration <= 16):
                continue

            if speaker in spk_utt_map:
                spk_utt_map[speaker] = spk_utt_map[speaker] + 1
            else:
                spk_utt_map[speaker] = 1

            if spk_utt_map[speaker] <= num_utt_spk:
                json.dump(jd, tm)
                tm.write("\n")
                print(f"speaker: {speaker}, count: {spk_utt_map[speaker]}")
