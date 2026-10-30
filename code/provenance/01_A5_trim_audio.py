# 01_A5_trim_audio
# AIM: Convert recordings to 16 kHz mono WAV, cut at start of speech (0.10 s lead), keep a 1.0 s tail, name files wes_/camf_NNNNN, write audio_preprocessing_log.csv
# saved: 2026-10-05 10:53

# ---------------------------------------------------------------
# A5 (user's version, tail = 1 s). Needs from earlier cells: TARGET_SR, TARGET_CHANNELS, PCM_SUBTYPE,
# LEAD_SEC, ONSET_DROP_DB, MIN_SPEECH_SEC, ID_PREFIX, AUDIO_OUT_DIR, RESULTS_DIR,
# item_number, pidgin_sources, camf_sources, pidgin_df, camf_df
TAIL_MIN_SEC = 1.0   # always end with 1 s after the voice (silence-padded if the source is shorter)
TAIL_MAX_SEC = 1.0   # never keep more than 1 s after the voice
# ---------------------------------------------------------------

subprocess.run(["bash", "-lc",
                "ffmpeg -version >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq ffmpeg)"],
               check=True)

def frame_db(y, sr, hop_ms=10, win_ms=25):
    hop = int(sr * hop_ms / 1000); win = int(sr * win_ms / 1000)
    n = 1 + max(0, (len(y) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    f = y[idx]
    return 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-9), hop

def speech_runs(mask, min_frames, max_gap):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    runs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > max_gap + 1:
            runs.append((s, p)); s = i
        p = i
    runs.append((s, p))
    return [(a, b) for a, b in runs if b - a + 1 >= min_frames]

def trim_speech(y, sr, lead, tail_min, tail_max, onset_drop, min_speech):
    y = y.astype(np.float32)
    db, hop = frame_db(y, sr)
    level = np.percentile(db, 95)                 # loud-speech level of THIS file
    floor = np.percentile(db, 10)                 # background-noise level of THIS file
    thr = max(level - onset_drop, floor + 12)
    runs = speech_runs(db > thr, int(min_speech * 100), 8)     # sound >= 150 ms (bridges gaps <= 80 ms)
    if not runs:
        return y, dict(speech_found=False, flags="no_speech_found")
    on, off = runs[0][0], runs[-1][1]
    thr_soft = max(floor + 10, thr - 12)
    while on > 0 and db[on - 1] > thr_soft and runs[0][0] - on < 15:   # quiet first consonant (max 150 ms back)
        on -= 1
    thr_end = max(level - onset_drop - 10, floor + 8)
    k = off
    while k + 1 < len(db) and db[k + 1] > thr_end and k - off < 40:     # soft word ending (up to 400 ms)
        k += 1
    start = max(0, on * hop - int(lead * sr))
    speech_end = min(len(y), (k + 1) * hop + int(0.025 * sr))
    avail = (len(y) - speech_end) / sr
    tail = min(tail_max, max(avail, 0.0))
    z = y[start: speech_end + int(tail * sr)]
    flags = []
    if tail < tail_min:
        z = np.concatenate([z, np.zeros(int(round((tail_min - tail) * sr)), np.float32)])
        flags.append("tail_padded_with_silence")
    if on * hop / sr < lead:
        flags.append("speech_starts_at_file_start")
    if len(runs) > 1:
        if runs[1][0] - runs[0][1] > 100 and runs[0][1] - runs[0][0] < 50:
            flags.append("isolated_noise_before_speech?")
        if runs[-1][0] - runs[-2][1] > 100 and runs[-1][1] - runs[-1][0] < 50:
            flags.append("isolated_noise_after_speech?")
    if len(z) / sr < 0.8:
        flags.append("very_short")
    if len(z) / sr > 15:
        flags.append("very_long")
    return z, dict(speech_found=True, speech_start_sec=on * hop / sr, speech_end_sec=speech_end / sr,
                   lead_removed_sec=start / sr, tail_kept_sec=max(tail, tail_min),
                   final_sec=len(z) / sr, flags=";".join(flags))

def process_variety(sources, df, variety):
    prefix = ID_PREFIX[variety]
    valid_ids = set(df["id"])
    out_dir = AUDIO_OUT_DIR / variety
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_decode_tmp.wav"
    rows = []
    for src in sources:
        n = item_number(src)
        item_id = f"{prefix}_{n:05d}"
        assert item_id in valid_ids, f"{item_id} is not in the CSV"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ac", str(TARGET_CHANNELS), "-ar", str(TARGET_SR), "-sample_fmt", "s16", str(tmp)],
                       check=True)
        y, sr = sf.read(tmp, dtype="float32")
        assert sr == TARGET_SR and y.ndim == 1
        z, info = trim_speech(y, sr, LEAD_SEC, TAIL_MIN_SEC, TAIL_MAX_SEC, ONSET_DROP_DB, MIN_SPEECH_SEC)
        sf.write(out_dir / f"{item_id}.wav", z, sr, subtype=PCM_SUBTYPE)
        rows.append(dict(variety=variety, id=item_id, source_file=src.name, source_sec=len(y) / sr,
                         speech_start_sec=info.get("speech_start_sec"), speech_end_sec=info.get("speech_end_sec"),
                         lead_removed_sec=info.get("lead_removed_sec"), tail_kept_sec=info.get("tail_kept_sec"),
                         final_sec=len(z) / sr, flags=info.get("flags", "")))
    tmp.unlink(missing_ok=True)
    return pd.DataFrame(rows)

audio_log = pd.concat([process_variety(pidgin_sources, pidgin_df, "pidgin"),
                       process_variety(camf_sources, camf_df, "camfranglais")], ignore_index=True)
audio_log.to_csv(RESULTS_DIR / "audio_preprocessing_log.csv", index=False)

for variety, prefix in ID_PREFIX.items():
    files_ = sorted((AUDIO_OUT_DIR / variety).glob("*.wav"))
    assert len(files_) == 114, f"{variety}: {len(files_)} processed files"
    for p in files_:
        info = sf.info(p)
        assert info.samplerate == TARGET_SR and info.channels == 1 and info.subtype == PCM_SUBTYPE, p.name
print("✅ 114 + 114 WAV files written (16 kHz, mono, 16-bit).")
print(audio_log.groupby("variety")[["source_sec", "lead_removed_sec", "final_sec"]].describe().T.round(2).to_string())
flagged = audio_log[audio_log["flags"] != ""]
print(f"\nFiles to listen to ({len(flagged)}):")
print(flagged[["variety", "id", "source_sec", "final_sec", "flags"]].round(2).to_string(index=False))
