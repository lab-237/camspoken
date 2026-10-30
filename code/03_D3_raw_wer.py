# 03_D3_raw_wer
# AIM: Raw (un-normalised) WER per model/variety/condition from saved transcripts; no Whisper rerun
# saved: 2026-10-06 02:05

import re, jiwer
t = pd.read_csv(RESULTS_DIR/"appendix_c_asr_transcripts.csv", keep_default_na=False)
rows = []
for (v, m, c), g in t.groupby(["variety", "model", "condition"]):
    ref = [re.sub(r"\s+", " ", x).strip() for x in g.reference_raw]
    hyp = [re.sub(r"\s+", " ", re.sub(r"<\|.*?\|>", "", x)).strip() for x in g.hypothesis_raw]
    rows.append(dict(variety=v, model=m, condition=c, raw_wer=round(100*jiwer.wer(ref, hyp), 1)))
raw = pd.DataFrame(rows)
raw.to_csv(RESULTS_DIR/"asr_raw_wer.csv", index=False)
print(raw.to_string(index=False))
save_code("03_D3_raw_wer", "Raw (un-normalised) WER per model/variety/condition from saved transcripts; no Whisper rerun")