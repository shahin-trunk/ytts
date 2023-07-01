import json

src = "data/manifest/manifest.json"
tgt = "data/manifest/manifest_orig_6_spk.json"

with open(src, encoding="utf-8") as sm:
    with open(tgt, encoding="utf-8", mode="w") as tm:
        for line in sm:
            jd = json.loads(line.strip("\n").strip())
            if jd["speaker"] not in ["6", "10"]:
                jd["ar_faraza"] = jd["text"]
                json.dump(jd, tm)
                tm.write("\n")
