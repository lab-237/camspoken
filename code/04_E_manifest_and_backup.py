# 04_E_manifest_and_backup
# AIM: Consistency checks (276 NLI runs, 2052 transcripts), audio checksums, run_manifest.json, results inventory, backup zip
# saved: 2026-10-05 23:23

# E: final manifest, inventory with SHA-256 checksums, consistency checks and a backup zip. CPU only, writes summary files only.
import importlib.metadata as md, zipfile as zf
_SKIP_EXT = (".gdoc", ".gsheet", ".gslides", ".gform", ".gdraw", ".gmap", ".gsite", ".gjam", ".gscript")
def _ok(p): return p.is_file() and not p.name.startswith(".") and p.suffix.lower() not in _SKIP_EXT  # skip Drive shortcut files
import datetime
from datetime import timezone

def sha(p): return sha256_file(p)

# ---- 1. consistency checks on the saved results ----
fr = pd.read_csv(RESULTS_DIR / "fold_results.csv")
key = ["experiment", "train_variety", "test_variety", "direction", "condition", "epochs", "seed", "fold"]
assert not fr.duplicated(key).any(), "duplicate NLI runs found"
expected = {"within_native": 30, "within_gloss": 30, "within_premise_only": 30, "within_hypothesis_only": 30,
            "epoch_sensitivity": 90, "transfer_disjoint": 60, "transfer_old_counterpart_leak": 6}
got = fr["experiment"].value_counts().to_dict()
assert got == expected, f"NLI run counts differ: {got}"
tr = pd.read_csv(RESULTS_DIR / "asr_transcripts_raw.csv", keep_default_na=False)
assert len(tr) == 2052 and tr.groupby(["model", "variety", "condition"]).size().eq(114).all(), "ASR transcript counts differ"
print(f"✅ NLI: {len(fr)} runs, no duplicates; ASR: {len(tr)} transcripts in 18 blocks of 114")

# ---- 2. audio checksums and statistics ----
rows = []
for v in ID_PREFIX:
    for p in sorted((AUDIO_OUT_DIR / v).glob("*.wav")):
        i = sf.info(p); rows.append(dict(variety=v, file=f"{v}/{p.name}", sha256=sha(p), seconds=round(i.duration, 3),
                                         samplerate=i.samplerate, channels=i.channels, subtype=i.subtype))
audio_ck = pd.DataFrame(rows); audio_ck.to_csv(RESULTS_DIR / "audio_checksums.csv", index=False)
assert len(audio_ck) == 228
dur = audio_ck.groupby("variety")["seconds"].agg(["min", "mean", "max"]).round(2)
print(dur.to_string())

# ---- 3. results and code inventory ----
inv = pd.DataFrame([dict(file=str(p.relative_to(DRIVE_ROOT)), bytes=p.stat().st_size, sha256=sha(p))
                    for d in ["results", "code", "final_csv"] for p in sorted((DRIVE_ROOT / d).rglob("*"))
                    if _ok(p) and p.name not in ("RESULTS_INVENTORY.csv", "run_manifest.json")])
inv.to_csv(RESULTS_DIR / "RESULTS_INVENTORY.csv", index=False)

# ---- 4. manifest ----
env = json.load(open(RESULTS_DIR / "environment_nli.json")) if (RESULTS_DIR / "environment_nli.json").exists() else {}
def ver(n):
    try: return md.version(n)
    except Exception: return None
manifest = dict(
    created_utc=datetime.datetime.now(timezone.utc).isoformat(timespec="seconds"),
    data=dict(files={n: sha(FINAL_CSV_DIR / n) for n in ["pidgin.csv", "camfranglais.csv"]}, rows_per_variety=114,
              label_balance="38/38/38", english_gloss_identical_across_varieties=True,
              note="Frozen after repair, tidy and gloss standardization; see code/01_A3*.py"),
    audio=dict(files=228, per_variety=114, format="16 kHz, mono, PCM 16-bit WAV", checksums="results/audio_checksums.csv",
               trim_rule=dict(lead_sec=LEAD_SEC, tail_sec=TAIL_MIN_SEC, onset_drop_db=ONSET_DROP_DB, min_speech_sec=MIN_SPEECH_SEC,
                              soft_onset_max_ms=150, soft_ending_max_ms=400, note="cut at start of speech; tail padded with silence if the source is shorter"),
               duration_sec={v: dur.loc[v].to_dict() for v in dur.index}),
    nli=dict(model=MODEL_NAME, max_length=MODEL_MAX_LENGTH, train_batch=TRAIN_BATCH_SIZE, eval_batch=EVAL_BATCH_SIZE, lr=LEARNING_RATE,
             weight_decay=WEIGHT_DECAY, optimizer=OPTIMIZER, lr_scheduler=LR_SCHEDULER, fp16=True, main_epochs=MAIN_EPOCHS,
             epoch_sensitivity=EPOCH_SENSITIVITY, seeds=MODEL_SEEDS, folds=N_FOLDS, fold_seed=FOLD_SEED,
             shared_folds_sha256=sha(RESULTS_DIR / "shared_item_folds.csv"), runs=len(fr), run_counts=expected,
             text_input="raw CSV text, no lowercasing or punctuation removal"),
    asr=dict(package="openai-whisper " + str(ver("openai-whisper")), models=["base", "small", "large-v3"],
             conditions=["auto", "forced_en", "forced_fr"], varieties=["pidgin", "camfranglais"], transcripts=len(tr),
             decoding="whisper defaults (task=transcribe, fp16), torch.manual_seed(0) before every clip",
             scoring_normalization="strip <|..|> tags; NFC; lowercase; curly apostrophes -> '; keep apostrophes; any other non-letter/digit/mark -> space; collapse spaces",
             metrics="WER, CER, MER, WIL, WIP (corpus level on normalized text)"),
    environment=dict(recorded_at_session_start=env, current_session_gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"),
                     packages={n: ver(n) for n in ["transformers", "datasets", "scikit-learn", "scipy", "numpy", "pandas", "torch", "openai-whisper", "soundfile", "librosa", "accelerate"]}),
    inventory="results/RESULTS_INVENTORY.csv")
json.dump(manifest, open(RESULTS_DIR / "run_manifest.json", "w"), indent=2)
print("manifest written; GPU recorded at session start:", env.get("gpu"))

# ---- 5. backup zip (results, code, final CSVs; audio is checksummed and packaged for Hugging Face separately) ----
(DRIVE_ROOT / "backup").mkdir(exist_ok=True)
zp = DRIVE_ROOT / "backup" / f"CamSpokenNLI_run1_results_code_data_{datetime.datetime.now():%Y%m%d}.zip"
with zf.ZipFile(zp, "w", zf.ZIP_DEFLATED) as z:
    for d in ["results", "code", "final_csv"]:
        for p in sorted((DRIVE_ROOT / d).rglob("*")):
            if _ok(p): z.write(p, p.relative_to(DRIVE_ROOT))
print(f"✅ backup zip: {zp.relative_to(DRIVE_ROOT)} ({zp.stat().st_size/1e6:.1f} MB), {len(inv)} files inventoried")

save_code("04_E_manifest_and_backup", "Consistency checks (276 NLI runs, 2052 transcripts), audio checksums, run_manifest.json, results inventory, backup zip")