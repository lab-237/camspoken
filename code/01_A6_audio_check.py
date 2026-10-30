# 01_A6_audio_check
# AIM: Verify processed audio on Drive (names, format, log match, 1 s tail) and play a sample
# saved: 2026-10-05 10:42

# A6: check the processed audio already on Drive (no changes) + play a sample. Also defines save_code().
import datetime, random
import IPython.display as ipd

def save_code(name, aim):
    src = get_ipython().user_ns["_ih"][-1]
    code_dir = DRIVE_ROOT / "code"; code_dir.mkdir(exist_ok=True)
    stamp = f"{datetime.datetime.now():%Y-%m-%d %H:%M}"
    (code_dir / f"{name}.py").write_text(f"# {name}\n# AIM: {aim}\n# saved: {stamp}\n\n" + src, encoding="utf-8")
    idx = code_dir / "INDEX.csv"
    old = pd.read_csv(idx) if idx.exists() else pd.DataFrame(columns=["name", "aim", "saved", "sha256"])
    old = old[old["name"] != name]
    row = pd.DataFrame([dict(name=name, aim=aim, saved=stamp, sha256=hashlib.sha256(src.encode()).hexdigest())])
    pd.concat([old, row], ignore_index=True).sort_values("name").to_csv(idx, index=False)
    print(f"📝 code saved: Drive/CamSpokenNLI_run1/code/{name}.py")

log = pd.read_csv(RESULTS_DIR / "audio_preprocessing_log.csv")
dfs = {"pidgin": pidgin_df, "camfranglais": camf_df}
for v, df in dfs.items():
    files_ = sorted((AUDIO_OUT_DIR / v).glob("*.wav"))
    assert {p.stem for p in files_} == set(df["id"]), f"{v}: file names do not match the CSV ids"
    for p in files_:
        i = sf.info(p)
        assert i.samplerate == TARGET_SR and i.channels == 1 and i.subtype == PCM_SUBTYPE, p.name
    lv = log[log["variety"] == v].set_index("id")
    dur = pd.Series({p.stem: sf.info(p).duration for p in files_})
    assert (abs(dur - lv["final_sec"].reindex(dur.index)) < 0.01).all(), f"{v}: files differ from the log (old run?)"
    print(f"✅ {v}: 114 files, names = CSV ids, 16 kHz mono 16-bit, durations match the log; "
          f"tail kept {lv['tail_kept_sec'].min():.2f}-{lv['tail_kept_sec'].max():.2f} s; "
          f"final {lv['final_sec'].min():.2f}-{lv['final_sec'].max():.2f} s")
print("clips padded with silence at the end:", int(log["flags"].fillna("").str.contains("tail_padded").sum()), "of", len(log))

rng = random.Random(1)
for v, df in dfs.items():
    lv = log[log["variety"] == v].sort_values("final_sec")
    ids = [lv.iloc[0]["id"], lv.iloc[-1]["id"]] + rng.sample(sorted(df["id"]), 4)
    text = df.set_index("id")["premise_text"]
    print(f"\n===== {v}: shortest, longest, 4 random =====")
    for i in ids:
        print(f"{i}: {text[i]}")
        ipd.display(ipd.Audio(str(AUDIO_OUT_DIR / v / f"{i}.wav")))

save_code("01_A6_audio_check", "Verify processed audio on Drive (names, format, log match, 1 s tail) and play a sample")