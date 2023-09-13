import json
import os
from pathlib import Path
from uuid import uuid4

data_path = "data/audio/multi_lang"
os.makedirs(data_path, exist_ok=True)

tgt_spk_map = "data/audio/multi_lang/speaker_map_hi.json"
tgt_spk_map_edited = "data/audio/multi_lang/speaker_map_new_id_hi.json"

lang_manifest_map = {
    # "ar": ["data/audio/multi_lang/manifest_gen_ar.json", "data/audio/multi_lang/manifest_org_ar.json"],
    # "en": ["data/audio/multi_lang/manifest_gen_en.json", "data/audio/multi_lang/manifest_org_en.json"],
    "hi": ["data/audio/multi_lang/manifest_org_hi.json"],
}

no_id_spk_map = {"ar": {"female1": "spk_65000",
                        "male1": "spk_65001",
                        "male2": "spk_65002",
                        "male3": "spk_65003",
                        "male4": "spk_65004",
                        "male5": "spk_65005",
                        "male6": "spk_65006",
                        "male7": "spk_65007", },
                 "en": {"female1": "spk_65008",
                        "male1": "spk_65009",
                        "male2": "spk_65010",
                        "male3": "spk_65011",
                        "male4": "spk_65012", }
                 }
spk_map = {}
for lang, manifests in lang_manifest_map.items():
    lang_spk_map = {}
    for manifest in manifests:
        with open(manifest, encoding="utf-8") as sm:
            for line in sm:
                jd = json.loads(line.strip("\n").strip())
                if "speaker" not in jd:
                    speaker = str(Path(jd["audio_filepath"]).parent).split("/")[-2]
                    if lang == "en" and speaker in ["male5", "male6"]:
                        continue
                    speaker = no_id_spk_map[lang][speaker]
                else:
                    speaker = jd["speaker"]

                lang_spk_map[str(speaker)] = {"id": f"{lang}_{speaker}",
                                              "count": lang_spk_map[str(speaker)]["count"] + 1 if str(
                                                  speaker) in lang_spk_map else 1}

    spk_map[lang] = lang_spk_map

print(spk_map)
with open(tgt_spk_map, mode="w", encoding="utf-8") as tsm:
    json.dump(spk_map, tsm)

with open(tgt_spk_map, encoding="utf-8") as tsm:
    spk_map = json.load(tsm)
    for lang, spk_data in spk_map.items():
        index = 0
        for org_spk_id, spk_meta in spk_data.items():
            index = index + 1
            spk_meta["id"] = f"{lang}_spk_{index}"
            spk_data[org_spk_id] = spk_meta

        spk_map[lang] = spk_data

with open(tgt_spk_map_edited, mode="w", encoding="utf-8") as tsm:
    json.dump(spk_map, tsm)

with open(tgt_spk_map_edited, encoding="utf-8") as tsm:
    spk_map = json.load(tsm)
    # print(spk_map["en"]["1272"])
    for lang, manifests in lang_manifest_map.items():
        lang_spk_map = {}
        for manifest in manifests:
            new_manifest = manifest.replace(".json", "_ns.json")
            with open(new_manifest, encoding="utf-8", mode="w") as tm:
                with open(manifest, encoding="utf-8") as sm:
                    for line in sm:
                        jd = json.loads(line.strip("\n").strip())
                        if "speaker" not in jd:
                            speaker = str(Path(jd["audio_filepath"]).parent).split("/")[-2]
                            if lang == "en" and speaker in ["male5", "male6"]:
                                continue
                            speaker = no_id_spk_map[lang][speaker]
                        else:
                            speaker = jd["speaker"]

                        jd["speaker"] = spk_map[lang][str(speaker)]["id"]
                        jd["u_fid"] = str(uuid4())
                        af = jd["audio_filepath"]
                        jd["audio_filepath"] = str(af).replace("//", "/")
                        json.dump(jd, tm)
                        tm.write("\n")

# char_set = set()
# words_en = set()
# words_hi = set()
# for lang, manifests in lang_manifest_map.items():
#     if lang not in ["ar"]:
#         for manifest in manifests:
#             new_manifest = manifest.replace(".json", "_ns.json")
#             write_json = new_manifest.replace(".json", "_clean.json")
#             with open(write_json, encoding="utf-8", mode="w") as tm:
#                 with open(new_manifest, encoding="utf-8") as sm:
#                     for line in sm:
#                         jd = json.loads(line.strip("\n").strip())
#                         text = str(jd["text"])
#
#                         text = text.replace("\t", " ").replace("\n", " ")
#                         text = text.replace("—", " - ")
#                         text = text.replace("–", " - ")
#                         text = text.replace('️', "")
#                         text = text.replace("[", " ")
#                         text = text.replace("]", " ")
#                         text = text.replace("  ", " ")
#
#                         if lang == "hi":
#
#                             text = text.replace("₹", " रूपया ")
#                             text = text.replace("  ", " ")
#
#                             for word in text.split():
#                                 words_hi.add(word)
#
#                         elif lang == "en":
#                             text = text.replace("  ", " ")
#                             for word in text.split():
#                                 words_en.add(word)
#
#                         jd["text"] = text
#                         json.dump(jd, tm)
#                         tm.write("\n")
#
#                         for ch in text:
#                             char_set.add(ch)
#
#             print(f"Done, lang: {lang}, manifest: {new_manifest}")

# char_set = list(sorted(char_set))
# print(char_set)
#
# words_en = list(sorted(words_en))
# print(words_en)
#
# words_hi = list(sorted(words_hi))
# print(words_hi)
