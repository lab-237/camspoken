# 00b_shared_folds
# AIM: Session start (step 2 of 3): NLI setup, environment record and the single shared item-level stratified 5-fold split
# saved: 2026-10-24 13:32

# 00b_shared_folds
# AIM: Session start (step 2 of 3): NLI setup, environment record and the single shared item-level stratified 5-fold split
# Run after 00a, in the same session (about 1 min).
# Wait for: the fold fingerprint 6cc2c87a...978f and the line 'One shared item-level stratified 5-fold split fixed'.

# Guard: stops with a clear message if 00a has not been run in this session.
assert "pidgin_df" in globals() and "LABEL2ID" in globals(), "Run 00a_session_and_data.py first, in the same session."

# B1: NLI setup + ONE shared 5-fold split (built from the Pidgin labels, copied to Camfranglais by id). Fixed for the whole run.
import gc, csv, time, math, glob, inspect, platform
import scipy, sklearn, transformers, datasets
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)
warnings.filterwarnings("ignore")

NUM_LABELS = 3
CHANCE_P = 1.0 / 3.0
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
TMP_TRAIN_DIR = Path("/content/camspoken_tmp_train"); TMP_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
assert torch.cuda.is_available(), "No GPU: set Runtime > Change runtime type > T4 GPU"
env = dict(gpu=GPU_NAME, python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
           datasets=datasets.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, numpy=np.__version__, pandas=pd.__version__)
json.dump(env, open(RESULTS_DIR / "environment_nli.json", "w"), indent=2)
print(env)
if "T4" not in GPU_NAME.upper():
    print("⚠️ Not a T4 - the protocol was fixed to T4; this is recorded in environment_nli.json")

# ---- shared folds ----
fold_source = pidgin_df[["id", "label"]].copy().sort_values("id").reset_index(drop=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
fold_values = np.full(len(fold_source), -1, dtype=int)
for fold_id, (_, test_idx) in enumerate(skf.split(fold_source["id"], fold_source["label"])):
    fold_values[test_idx] = fold_id
assert np.all(fold_values >= 0)
fold_map = dict(zip(fold_source["id"], fold_values))      # keys are Pidgin ids (wes_NNNNN)
num = lambda s: s.str[-5:]                                  # match Camfranglais to Pidgin by item number
fold_by_num = {k[-5:]: v for k, v in fold_map.items()}
pidgin_df["fold"] = pidgin_df["id"].map(fold_map).astype(int)
camf_df["fold"] = num(camf_df["id"]).map(fold_by_num).astype(int)

assert (num(pidgin_df["id"]).values == num(camf_df["id"]).values).all()
assert (pidgin_df["label"].values == camf_df["label"].values).all()
assert (pidgin_df["fold"].values == camf_df["fold"].values).all()

fold_table = pidgin_df[["id", "label", "fold"]].copy()
fold_table.insert(1, "camf_id", camf_df["id"].values)
fold_table.to_csv(RESULTS_DIR / "shared_item_folds.csv", index=False)

print("\nItems per fold and label:")
print(pd.crosstab(pidgin_df["fold"], pidgin_df["label"]))
for label in LABEL2ID:
    counts = pidgin_df.loc[pidgin_df["label"] == label, "fold"].value_counts().sort_index()
    assert counts.max() - counts.min() <= 1, f"Fold imbalance for {label}: {counts.to_dict()}"
print("\nTest-set sizes per fold:", pidgin_df["fold"].value_counts().sort_index().tolist())
print("shared_item_folds.csv SHA-256:", sha256_file(RESULTS_DIR / "shared_item_folds.csv"))
print(f"✅ One shared item-level stratified {N_FOLDS}-fold split fixed (fold seed {FOLD_SEED}).")

save_code("00b_shared_folds", "Session start (step 2 of 3): NLI setup, environment record and the single shared item-level stratified 5-fold split")