import json

num_samples = 25

src = "data/audio/multi_lang/manifest_org_ar_ns_phoneme.json"
tgt = f"data/audio/multi_lang/manifest_org_ar_ns_phoneme_sd_{num_samples}.json"

spk_map = {}

with open(tgt, encoding="utf-8", mode="w") as tm:
    with open(src, encoding="utf-8") as sm:
        for line in sm:
            jd = json.loads(line.strip("\n").strip())
            speaker = jd["speaker"]
            if speaker not in spk_map:
                spk_map[speaker] = 1
                json.dump(jd, tm)
                tm.write("\n")
            else:
                if int(spk_map[speaker]) < num_samples:
                    spk_map[speaker] = spk_map[speaker] + 1
                    json.dump(jd, tm)
                    tm.write("\n")
