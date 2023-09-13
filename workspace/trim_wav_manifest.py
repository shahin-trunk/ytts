import json
import logging
import os

import torch
import torchaudio
import torchaudio.functional as taf

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("TWM")

src = "data/audio/multi_lang/manifest_org_en_ns_clean.json"
tgt = "data/audio/multi_lang/manifest_org_en_ns_clean_trim.json"


def trim_silence(source: str, target: str = None, sample_rate: int = 22050, resample: bool = True):
    if target is None:
        target = source.replace(".wav", "_trim.wav").replace("//", "/")
        if os.path.exists(target):
            return target
    try:
        aud, sr = torchaudio.load(source)
        aud = aud.to(torch.device('cuda:0'))
        if aud.shape[0] > 1:
            aud = aud.mean(aud, dim=0, keepdim=True)

        if resample:
            aud = taf.resample(aud, sr, sample_rate)

        aud = taf.vad(aud, sample_rate)

        aud, sr = torchaudio.sox_effects.apply_effects_tensor(aud.to(torch.device('cpu')), sample_rate, [['reverse']])
        aud = taf.vad(aud.to(torch.device('cuda:0')), sample_rate)
        aud, sr = torchaudio.sox_effects.apply_effects_tensor(aud.to(torch.device('cpu')), sample_rate, [['reverse']])
        torchaudio.save(target, aud, sample_rate, bits_per_sample=32)
        return target
    except Exception as et:
        raise et


with open(tgt, encoding="utf-8", mode="w") as tm:
    with open(src, encoding="utf-8") as sm:
        count = 0
        for index, line in enumerate(sm):
            # if index <= 958700:
            #     continue
            try:
                jd = json.loads(line.strip("\n").strip())
                src_af = str(jd["audio_filepath"])
                tgt_af = trim_silence(src_af)
                jd["audio_filepath"] = tgt_af
                audio_meta = torchaudio.info(tgt_af)
                duration = audio_meta.num_frames / audio_meta.sample_rate
                jd["duration"] = duration
                json.dump(jd, tm)
                tm.write("\n")
                count = count + 1

                if count % 100 == 0:
                    _log.info(f"Done: {count}")
            except Exception as e:
                _log.error(e)
                raise e
