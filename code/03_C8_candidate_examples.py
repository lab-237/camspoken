# 03_C8_candidate_examples
# AIM: List non-Latin and zero-error Large-v3 transcripts as candidate examples for Appendix C.8
# saved: 2026-10-06 23:26

import unicodedata, re
t = pd.read_csv(RESULTS_DIR/"appendix_c_asr_transcripts.csv", keep_default_na=False)
t = t[t.model=="large-v3"].copy()
t["wer_clip"] = ((t.substitutions + t.deletions + t.insertions) / t.reference_words.replace(0, 1) * 100).round(0)
def nonlatin(s):
    return any(ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN") for ch in str(s))
auto = t[t.condition=="auto"].set_index("id")
forced = {v: t[(t.variety==v) & (t.condition==("forced_en" if v=="pidgin" else "forced_fr"))].set_index("id") for v in ["pidgin","camfranglais"]}
for v in ["pidgin","camfranglais"]:
    a = auto[auto.variety==v]
    print(f"\n===== {v}: Large-v3 automatic transcripts with non-Latin letters =====")
    for i, r in a[a.hypothesis_raw.map(nonlatin)].iterrows():
        f = forced[v].loc[i]
        print(f"{i} | ref: {r.reference_raw}\n   auto [{r.detected_language}] WER {r.wer_clip:.0f}%: {r.hypothesis_raw}\n   forced WER {f.wer_clip:.0f}%: {f.hypothesis_raw}")
print("\n===== Camfranglais clips with zero errors, Large-v3, automatic =====")
for i, r in auto[(auto.variety=="camfranglais") & (auto.substitutions+auto.deletions+auto.insertions==0)].iterrows():
    print(i, "|", r.reference_raw, "->", r.hypothesis_raw, f"[{r.detected_language}]")
save_code("03_C8_candidate_examples", "List non-Latin and zero-error Large-v3 transcripts as candidate examples for Appendix C.8")