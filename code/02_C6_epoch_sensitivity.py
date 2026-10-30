# 02_C6_epoch_sensitivity
# AIM: Epoch sensitivity (5, 10, 20 new; 15 from main run), native text, within-variety CV, 3 seeds
# saved: 2026-10-05 17:10

# C6: epoch sensitivity - native text, within-variety 5-fold, 3 seeds: NEW runs at 5, 10 and 20 epochs.
# The 15-epoch values come from the main run (C3). All settings are reported side by side; none is selected as a winner.
# 2 varieties x 3 epoch settings x 15 runs = 90 runs (about 70 min). Resumable.
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
datasets.disable_progress_bar()

def run_within(variety, condition, epochs=MAIN_EPOCHS, experiment_name="within"):
    df = pidgin_df if variety == "pidgin" else camf_df
    existing = existing_result_keys()
    for seed in MODEL_SEEDS:
        for fold_id in range(N_FOLDS):
            train_df = df[df["fold"] != fold_id].copy().reset_index(drop=True)
            test_df = df[df["fold"] == fold_id].copy().reset_index(drop=True)
            assert set(train_df["id"]).isdisjoint(set(test_df["id"])) and len(train_df) + len(test_df) == 114
            row_meta = {"experiment": experiment_name, "split_type": "within_5fold", "direction": "within",
                        "train_variety": variety, "test_variety": variety, "condition": condition,
                        "epochs": epochs, "seed": seed, "fold": fold_id}
            key = result_key(row_meta)
            if key in existing:
                print("SKIP completed:", key); continue
            started = time.time()
            correct, n_test, acc, preds = run_training_with_retry(
                train_df, test_df, condition, seed, epochs, f"within_{variety}_{condition}_e{epochs}_s{seed}_f{fold_id}")
            append_row(FOLD_RESULTS_FILE, {**row_meta, "correct": correct, "n_test": n_test, "accuracy": acc,
                                           "fold_mean_component": acc, "status": "ok",
                                           "seconds": time.time() - started, "counterpart_overlap": 0}, RESULT_COLUMNS)
            append_prediction_rows([{**{k: row_meta[k] for k in ["experiment", "split_type", "direction", "train_variety",
                                                                "test_variety", "condition", "epochs", "seed", "fold"]}, **p} for p in preds])
            print(f"✅ {key} | {correct}/{n_test} = {acc:.4f}")

for variety in ["pidgin", "camfranglais"]:
    for ep in [5, 10, 20]:
        run_within(variety, "native_full", ep, "epoch_sensitivity")

fr = pd.read_csv(FOLD_RESULTS_FILE)
w = fr[(fr["condition"] == "native_full") & (fr["split_type"] == "within_5fold") &
       (fr["experiment"].isin(["within_native", "epoch_sensitivity"]))]
per_seed = w.groupby(["train_variety", "epochs", "seed"]).apply(lambda g: g["correct"].sum() / g["n_test"].sum()).rename("acc").reset_index()
summ = per_seed.groupby(["train_variety", "epochs"])["acc"].agg(["mean", "std", "count"]).round(4)
print("\nEpoch sensitivity, native text, within-variety (mean over seeds, sample SD; count = seeds):")
print(summ.to_string())
print("runs completed (5/10/20 epochs):", len(fr[fr["experiment"] == "epoch_sensitivity"]), "of 90")

save_code("02_C6_epoch_sensitivity", "Epoch sensitivity (5, 10, 20 new; 15 from main run), native text, within-variety CV, 3 seeds")