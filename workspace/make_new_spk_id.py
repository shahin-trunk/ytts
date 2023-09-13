import json

src = "data/audio/multi_lang/sel/manifest_ar_os_v1.json"
tgt = "data/audio/multi_lang/sel/manifest_ar_ns_v1.json"
spk = "data/audio/multi_lang/sel/spk_change_id_map.json"

spk_map = {}
spk_id_index = 0
lang = "ar"
with open(src, encoding="utf-8") as sm:
    with open(tgt, encoding="utf-8", mode="w") as tm:
        for line in sm:
            jd = json.loads(line.strip("\n").strip())
            speaker = jd["speaker"]
            duration = float(jd["duration"])
            if not (2.0 <= duration <= 16.0):
                continue

            if speaker in spk_map:
                jd["speaker"] = spk_map[speaker]["new_id"]
            else:
                spk_id_index = spk_id_index + 1
                jd["speaker"] = f"spk_{lang}_{spk_id_index}"
                spk_map[speaker] = {"index": spk_id_index, "new_id": jd["speaker"], "last_id": speaker}

            json.dump(jd, tm)
            tm.write("\n")

with open(spk, encoding="utf-8", mode="w") as spk_j:
    json.dump(spk_map, spk_j)
