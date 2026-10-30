# 01_A8_content_check
# AIM: Quick Whisper-small pass: does every processed clip match its own premise sentence (not a reported result)
# saved: 2026-10-05 11:01

# A8: content check - does each clip say its OWN sentence? (quick Whisper-small pass; not a reported result)
import unicodedata, difflib, whisper

def norm(s):
    s = re.sub(r"<\|[^|]*\|>", "", s)
    s = unicodedata.normalize("NFC", s).lower().replace("’", "'").replace("‘", "'")
    s = "".join(ch if (ch.isalnum() or unicodedata.category(ch).startswith("M") or ch == "'") else " " for ch in s)
    return " ".join(s.split())

LANG = {"pidgin": "en", "camfranglais": "fr"}            # only to help the check read the clips
OUT = RESULTS_DIR / "content_check_transcripts.csv"
done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["variety", "id", "transcript"])
have = set(done["id"])
todo = [(v, i) for v, df in (("pidgin", pidgin_df), ("camfranglais", camf_df)) for i in df["id"] if i not in have]
if todo:
    model = whisper.load_model("small", device="cuda" if torch.cuda.is_available() else "cpu")
    new = []
    for v, i in todo:
        r = model.transcribe(str(AUDIO_OUT_DIR / v / f"{i}.wav"), language=LANG[v], temperature=0,
                             condition_on_previous_text=False, fp16=torch.cuda.is_available())
        new.append(dict(variety=v, id=i, transcript=r["text"].strip()))
    done = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
    done.to_csv(OUT, index=False)
    del model; torch.cuda.empty_cache()

rows = []
for v, df in (("pidgin", pidgin_df), ("camfranglais", camf_df)):
    ref = {i: norm(t) for i, t in zip(df["id"], df["premise_text"])}
    text = dict(zip(df["id"], df["premise_text"]))
    for _, r in done[done["variety"] == v].iterrows():
        hyp = norm(str(r["transcript"]) if pd.notna(r["transcript"]) else "")
        sc = {i: difflib.SequenceMatcher(None, hyp, t).ratio() for i, t in ref.items()}
        best = max(sc, key=sc.get)
        rows.append(dict(variety=v, id=r["id"], own_sim=round(sc[r["id"]], 3), best_id=best,
                         best_sim=round(sc[best], 3), transcript=r["transcript"],
                         own_text=text[r["id"]], best_text=text[best]))
chk = pd.DataFrame(rows)
chk["flag"] = np.where((chk["best_id"] != chk["id"]) & (chk["best_sim"] - chk["own_sim"] > 0.05), "OTHER_SENTENCE_FITS_BETTER",
               np.where(chk["own_sim"] < 0.30, "VERY_LOW_MATCH", ""))
chk.to_csv(RESULTS_DIR / "content_check.csv", index=False)

print(chk.groupby("variety").agg(clips=("id", "size"), own_is_best=("own_sim", lambda s: int((chk.loc[s.index, "id"] == chk.loc[s.index, "best_id"]).sum())),
                                 median_own_sim=("own_sim", "median"), flagged=("flag", lambda s: int((s != "").sum()))).to_string())
bad = chk[chk["flag"] != ""]
print(f"\nClips to listen to ({len(bad)}):")
for _, r in bad.iterrows():
    print(f"\n{r['id']} [{r['flag']}] own={r['own_sim']} best={r['best_id']}({r['best_sim']})\n  heard : {r['transcript']}\n  own   : {r['own_text']}\n  best  : {r['best_text']}")

save_code("01_A8_content_check", "Quick Whisper-small pass: does every processed clip match its own premise sentence (not a reported result)")