# 03_D_whisper_asr
# AIM: Zero-shot Whisper Base/Small/Large-v3 on trimmed premise audio; auto, forced English and forced French for both varieties; WER/CER/MER/WIL/WIP
# saved: 2026-10-05 22:03

# D: Whisper zero-shot ASR (Base, Small, Large-v3) on the 114 trimmed premise clips per variety.
# Conditions for BOTH varieties: auto (language detected), forced_en, forced_fr. Default Whisper decoding, seed 0 before every clip.
# Scoring: premise_text is the reference; the corrected normalizer is used ONLY for scoring (the CSV text is never changed).
# About 1 hour on the T4 (Large-v3 last). Resumable: finished (model, variety, condition) blocks are skipped.
import unicodedata, whisper
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

def normalize_wer_text(text):
    t = "" if (text is None or (isinstance(text, float) and np.isnan(text))) else str(text)
    t = re.sub(r"<\|[^|]*\|>", "", t)                       # Whisper language/task tags
    t = unicodedata.normalize("NFC", t).lower()
    for a in ("\u2019", "\u2018", "\u02bc", "`", "\u00b4"):  # curly/modifier apostrophes -> straight
        t = t.replace(a, "'")
    out = [ch if (ch.isalnum() or ch == "'" or unicodedata.category(ch).startswith("M")) else " " for ch in t]
    return " ".join("".join(out).split())

def alignment_counts(ref_words, hyp_words):
    n, m = len(ref_words), len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    back = np.empty((n + 1, m + 1), dtype=object)

    for i in range(1, n + 1):
        dp[i, 0] = i
        back[i, 0] = "D"
    for j in range(1, m + 1):
        dp[0, j] = j
        back[0, j] = "I"
    back[0, 0] = ""

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = dp[i - 1, j - 1]
                op = "H"
            else:
                choices = [
                    (dp[i - 1, j - 1] + 1, "S"),
                    (dp[i - 1, j] + 1, "D"),
                    (dp[i, j - 1] + 1, "I"),
                ]
                cost, op = min(choices, key=lambda x: (x[0], {"S": 0, "D": 1, "I": 2}[x[1]]))
            dp[i, j] = cost
            back[i, j] = op

    i, j = n, m
    hits = subs = dels = ins = 0
    while i > 0 or j > 0:
        op = back[i, j]
        if op == "H":
            hits += 1; i -= 1; j -= 1
        elif op == "S":
            subs += 1; i -= 1; j -= 1
        elif op == "D":
            dels += 1; i -= 1
        elif op == "I":
            ins += 1; j -= 1
        else:
            raise RuntimeError("Invalid alignment state")
    return hits, subs, dels, ins

def counts_to_metrics(h, s, d, i, n_hyp):
    n_ref = h + s + d
    wip = (h / n_ref) * (h / n_hyp) if n_ref and n_hyp else 0.0
    return dict(error_rate=(s + d + i) / n_ref, mer=(s + d + i) / (h + s + d + i), wip=wip, wil=1 - wip)

# ---- self-test of the normalizer (the old one deleted the letters u,0,1,2,7,8,9) ----
assert normalize_wer_text("Don\u2019t stop,  please!") == "don't stop please"
assert normalize_wer_text("u 0127 89 <|en|> L'eau-chaude.") == "u 0127 89 l'eau chaude"
print("normalizer self-test passed")

CONDITIONS = {"auto": None, "forced_en": "en", "forced_fr": "fr"}
VARIETIES = [("pidgin", pidgin_df), ("camfranglais", camf_df)]
MODELS = ["base", "small", "large-v3"]
TRANS_FILE = RESULTS_DIR / "asr_transcripts_raw.csv"
TCOLS = ["variety", "model", "condition", "id", "audio_path", "reference_raw", "hypothesis_raw", "detected_language"]
trans = pd.read_csv(TRANS_FILE, keep_default_na=False) if TRANS_FILE.exists() else pd.DataFrame(columns=TCOLS)

def n_done(m, v, c):
    return int(((trans["model"] == m) & (trans["variety"] == v) & (trans["condition"] == c)).sum())

for m in MODELS:
    todo = [(v, df, c, l) for v, df in VARIETIES for c, l in CONDITIONS.items() if n_done(m, v, c) < 114]
    if not todo:
        print(f"SKIP {m}: all 6 blocks complete"); continue
    try:
        model = whisper.load_model(m, device="cuda")
    except Exception as e:
        print(f"❗ {m} could not be loaded ({e!r}); recorded as unavailable, no substitute used."); continue
    for v, df, c, lang in todo:
        t0 = time.time(); rows = []
        trans = trans[~((trans["model"] == m) & (trans["variety"] == v) & (trans["condition"] == c))]   # drop a partial block
        for _, r in df.sort_values("id").iterrows():
            p = AUDIO_OUT_DIR / v / f"{r['id']}.wav"
            assert p.exists(), p
            opts = {"task": "transcribe", "fp16": True}
            if lang: opts["language"] = lang
            torch.manual_seed(0)
            res = model.transcribe(str(p), **opts)
            rows.append(dict(variety=v, model=m, condition=c, id=r["id"], audio_path=f"processed_audio/{v}/{p.name}",
                             reference_raw=r["premise_text"], hypothesis_raw=res["text"].strip(),
                             detected_language=res.get("language", lang)))
        trans = pd.concat([trans, pd.DataFrame(rows)[TCOLS]], ignore_index=True)
        trans.to_csv(TRANS_FILE, index=False)
        print(f"✅ {m} / {v} / {c}: 114 clips in {time.time() - t0:.0f}s")
    del model; gc.collect(); torch.cuda.empty_cache()

# ---- scoring ----
trans = pd.read_csv(TRANS_FILE, keep_default_na=False)
trans["reference_norm"] = trans["reference_raw"].map(normalize_wer_text)
trans["hypothesis_norm"] = trans["hypothesis_raw"].map(normalize_wer_text)
clip_counts = [alignment_counts(r.split(), h.split()) for r, h in zip(trans["reference_norm"], trans["hypothesis_norm"])]
trans[["hits", "substitutions", "deletions", "insertions"]] = pd.DataFrame(clip_counts, index=trans.index)
trans["reference_words"] = trans["reference_norm"].str.split().str.len()
trans["hypothesis_words"] = trans["hypothesis_norm"].str.split().str.len()
trans.to_csv(RESULTS_DIR / "appendix_c_asr_transcripts.csv", index=False)

summ = []
for (v, m, c), g in trans.groupby(["variety", "model", "condition"]):
    H, S, D, I = (int(g[k].sum()) for k in ["hits", "substitutions", "deletions", "insertions"])
    met = counts_to_metrics(H, S, D, I, int(g["hypothesis_words"].sum()))
    cc = [alignment_counts(list(r), list(h)) for r, h in zip(g["reference_norm"], g["hypothesis_norm"])]
    cH, cS, cD, cI = (sum(x[k] for x in cc) for k in range(4))
    summ.append(dict(variety=v, model=m, condition=c, n_recordings=len(g), wer=met["error_rate"], cer=(cS + cD + cI) / (cH + cS + cD),
                     mer=met["mer"], wil=met["wil"], wip=met["wip"], reference_words=int(g["reference_words"].sum()),
                     hypothesis_words=int(g["hypothesis_words"].sum()), hits=H, substitutions=S, deletions=D, insertions=I,
                     language_tags="|".join(f"{k}:{n}" for k, n in g["detected_language"].value_counts().items())))
summ = pd.DataFrame(summ)
summ.to_csv(RESULTS_DIR / "asr_summary.csv", index=False)

wide = summ.pivot_table(index=["variety", "condition"], columns="model", values="wer").reindex(columns=MODELS) * 100
wide.to_csv(RESULTS_DIR / "table2_asr_wer_all_conditions.csv")
print("\nWER % (normalized, Whisper zero-shot):")
print(wide.round(1).to_string())
print("\nCER % :"); print((summ.pivot_table(index=["variety", "condition"], columns="model", values="cer").reindex(columns=MODELS) * 100).round(1).to_string())
print("\nLanguage tags in the AUTO condition:")
print(summ[summ["condition"] == "auto"][["variety", "model", "language_tags"]].to_string(index=False))
missing = 18 - len(summ)
print(f"\n{'✅ all 18 model/variety/condition blocks scored' if missing == 0 else f'⚠️ {missing} blocks missing (see messages above)'}")

save_code("03_D_whisper_asr", "Zero-shot Whisper Base/Small/Large-v3 on trimmed premise audio; auto, forced English and forced French for both varieties; WER/CER/MER/WIL/WIP")