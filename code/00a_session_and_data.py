# 00a_session_and_data
# AIM: Session start (step 1 of 3): install packages, mount Drive, set paths and settings, load the frozen CSVs (hash-checked)
# saved: 2026-10-24 18:16

# 00a_session_and_data
# AIM: Session start (step 1 of 3): install packages, mount Drive, set paths and settings, load the frozen CSVs (hash-checked)
# Run this first after every Colab reset (about 2 min), then 00b and 00c in the same session.
# Wait for: the GPU line and 'frozen CSVs verified'.

# Installs packages, mounts Drive, sets the config and loads the frozen CSVs (hash-checked).

import subprocess, sys, os, re, json, zipfile, shutil, hashlib, random, warnings
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "transformers==5.16.1", "datasets==4.8.5", "scikit-learn==1.6.1", "scipy==1.16.3",
                "openai-whisper==20250625", "jiwer==4.0.0", "librosa", "soundfile", "accelerate", "sentencepiece"],
               check=True)

import numpy as np, pandas as pd, soundfile as sf, torch
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# ---- paths (Drive = permanent) ----
DRIVE_ROOT    = Path("/content/drive/MyDrive/CamSpokenNLI_run1")
INPUT_DIR     = DRIVE_ROOT / "inputs"
FINAL_CSV_DIR = DRIVE_ROOT / "final_csv"
RESULTS_DIR   = DRIVE_ROOT / "results"
AUDIO_OUT_DIR = DRIVE_ROOT / "audio"
for d in (RESULTS_DIR, AUDIO_OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- NLI config ----
MODEL_NAME = "xlm-roberta-base"; MODEL_MAX_LENGTH = 64
TRAIN_BATCH_SIZE = 4; EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5; WEIGHT_DECAY = 0.0
OPTIMIZER = "adamw_torch"; LR_SCHEDULER = "linear"
MAIN_EPOCHS = 15; EPOCH_SENSITIVITY = [5, 10, 15, 20]
MODEL_SEEDS = [42, 123, 2024]; FOLD_SEED = 42; N_FOLDS = 5
LABEL2ID = {"entailment": 0, "contradiction": 1, "neutral": 2}
ID_PREFIX = {"pidgin": "wes", "camfranglais": "camf"}

# ---- audio config ----
TARGET_SR = 16000; TARGET_CHANNELS = 1; PCM_SUBTYPE = "PCM_16"
LEAD_SEC = 0.10; ONSET_DROP_DB = 18; MIN_SPEECH_SEC = 0.15
TAIL_MIN_SEC = 1.0; TAIL_MAX_SEC = 1.0

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (set Runtime > T4 GPU)")

# ---- frozen CSVs, hash-checked ----
FROZEN = {"pidgin.csv": "8e1886064798198372270886ad584fbdacf3c410a88eca6c0eb932c1f7ca7029",
          "camfranglais.csv": "85e984a30f88151b23561a6975609a2011675de45cb3ca15afcbe8dafe0d2300"}
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
for name, want in FROZEN.items():
    assert sha256_file(FINAL_CSV_DIR / name) == want, f"{name} does not match the frozen fingerprint"
pidgin_df = pd.read_csv(FINAL_CSV_DIR / "pidgin.csv")
camf_df   = pd.read_csv(FINAL_CSV_DIR / "camfranglais.csv")
assert len(pidgin_df) == len(camf_df) == 114
print("✅ frozen CSVs verified (114 rows each)")

# ---- code log: saves the code of the cell that calls it, plus its aim, into Drive/code (+ INDEX.csv) ----
import datetime
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


save_code("00a_session_and_data", "Session start (step 1 of 3): install packages, mount Drive, set paths and settings, load the frozen CSVs (hash-checked)")