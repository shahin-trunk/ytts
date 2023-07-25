import json
import logging
import os

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("RM")

src = "manifest_se_17072023_32b_1_final.json"
tgt = "manifest_se_17072023_32b_1_final.txt"

with open(src, encoding="utf-8") as sm:
    with open(tgt, encoding="utf-8", mode="w") as tt:
        for af_index, line in enumerate(sm):
            try:
                jd = json.loads(line.strip("\n").strip())
                text = str(jd['text']).strip("\n").strip()
                af = str(jd['audio_filepath'])
                os.makedirs(os.path.dirname(af), exist_ok=True)
                tt.write(f"{af}: {text}\n")
            except Exception as e:
                _log.error(str(e))
