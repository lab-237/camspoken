# 03_D2_asr_checks
# AIM: ASR checks: bootstrap CIs for WER, non-Latin/empty transcript rates, identical-value check, example transcripts
# saved: 2026-10-05 23:10

# D2: ASR checks for the paper - 95% bootstrap CIs, non-Latin-script and empty-output rates, identical-value check, example transcripts.
# Reads results/appendix_c_asr_transcripts.csv (written by D). No Whisper run, takes seconds.
import unicodedata, random
tr = pd.read_csv(RESULTS_DIR / "appendix_c_asr_transcripts.csv", keep_default_na=False)
tr["N"] = tr["hits"] + tr["substitutions"] + tr["deletions"]
MODELS = ["base", "small", "large-v3"]

# ---- 1. WER with 95% CI (resampling the 114 clips, 2000 draws, seed 0) ----
rng = np.random.default_rng(0)
cells, ci_rows = {}, []
for (v, c, m), g in tr.groupby(["variety", "condition", "model"]):
    g = g.sort_values("id")
    err = (g["substitutions"] + g["deletions"] + g["insertions"]).to_numpy(); n = g["N"].to_numpy()
    idx = rng.integers(0, len(g), (2000, len(g)))
    boot = err[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ci_rows.append(dict(variety=v, condition=c, model=m, wer=err.sum() / n.sum(), ci_low=lo, ci_high=hi))
    cells[(v, c, m)] = f"{100*err.sum()/n.sum():.1f} [{100*lo:.1f}-{100*hi:.1f}]"
pd.DataFrame(ci_rows).to_csv(RESULTS_DIR / "asr_wer_bootstrap_ci.csv", index=False)
tab = pd.DataFrame({m: {(v, c): cells[(v, c, m)] for (v, c, mm) in cells if mm == m} for m in MODELS})
print("WER % [95% bootstrap CI over clips]:"); print(tab.to_string())

# ---- 2. non-Latin script and empty outputs ----
def non_latin(s):
    return any(ch.isalpha() and "LATIN" not in unicodedata.name(ch, "") for ch in s)
tr["non_latin"] = tr["hypothesis_raw"].map(non_latin)
tr["empty"] = tr["hypothesis_norm"].str.strip().eq("")
sc = tr.groupby(["variety", "condition", "model"])[["non_latin", "empty"]].sum().astype(int).reset_index()
sc.to_csv(RESULTS_DIR / "asr_script_and_empty_counts.csv", index=False)
print("\nTranscripts (of 114) containing non-Latin letters / empty after normalization:")
print(sc.pivot_table(index=["variety", "condition"], columns="model", values=["non_latin", "empty"]).reindex(columns=MODELS, level=1).to_string())

# ---- 3. are small and large-v3 really identical for Pidgin forced French? ----
chk = tr[(tr["variety"] == "pidgin") & (tr["condition"] == "forced_fr") & (tr["model"].isin(["small", "large-v3"]))]
print("\nPidgin forced_fr counts (hits, substitutions, deletions, insertions):")
print(chk.groupby("model")[["hits", "substitutions", "deletions", "insertions"]].sum().to_string())
a = chk[chk["model"] == "small"].set_index("id")["hypothesis_raw"]; b = chk[chk["model"] == "large-v3"].set_index("id")["hypothesis_raw"]
print("identical transcripts between small and large-v3:", int((a == b.reindex(a.index)).sum()), "of", len(a))

# ---- 4. example transcripts (6 random clips per variety, seed 1; large-v3 auto vs large-v3 best forced language) ----
best = {"pidgin": "forced_en", "camfranglais": "forced_fr"}
ex = []
for v in ["pidgin", "camfranglais"]:
    L = tr[(tr["variety"] == v) & (tr["model"] == "large-v3")]
    for i in random.Random(1).sample(sorted(L["id"].unique()), 6):
        r = L[L["id"] == i]
        ex.append(dict(variety=v, id=i, reference=r["reference_raw"].iloc[0],
                       auto=r[r["condition"] == "auto"]["hypothesis_raw"].iloc[0],
                       auto_lang=r[r["condition"] == "auto"]["detected_language"].iloc[0],
                       best_forced=r[r["condition"] == best[v]]["hypothesis_raw"].iloc[0]))
ex = pd.DataFrame(ex); ex.to_csv(RESULTS_DIR / "asr_examples_large_v3.csv", index=False)
print("\nExamples (large-v3):")
for _, r in ex.iterrows():
    print(f"\n{r['id']}\n  ref : {r['reference']}\n  auto [{r['auto_lang']}]: {r['auto']}\n  forced ({best[r['variety']]}): {r['best_forced']}")

save_code("03_D2_asr_checks", "ASR checks: bootstrap CIs for WER, non-Latin/empty transcript rates, identical-value check, example transcripts")