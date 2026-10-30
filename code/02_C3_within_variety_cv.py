# 02_C3_within_variety_cv
# AIM: Within-variety 5-fold CV (3 seeds, 15 epochs): native, English gloss, premise-only, hypothesis-only
# saved: 2026-10-05 13:30

# C3: within-variety 5-fold CV, 3 seeds, 15 epochs - native text, English gloss, premise-only and hypothesis-only baselines.
# 2 varieties x 4 conditions x 3 seeds x 5 folds = 120 runs (about 1 to 1.5 h on the T4). Resumable: finished runs are skipped.
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

            train_items = set(train_df["id"])
            test_items = set(test_df["id"])
            assert train_items.isdisjoint(test_items)
            assert len(train_items) + len(test_items) == 114

            row_meta = {
                "experiment": experiment_name,
                "split_type": "within_5fold",
                "direction": "within",
                "train_variety": variety,
                "test_variety": variety,
                "condition": condition,
                "epochs": epochs,
                "seed": seed,
                "fold": fold_id,
            }
            key = result_key(row_meta)
            if key in existing:
                print("SKIP completed:", key)
                continue

            started = time.time()
            correct, n_test, acc, preds = run_training_with_retry(
                train_df, test_df, condition, seed, epochs,
                f"within_{variety}_{condition}_e{epochs}_s{seed}_f{fold_id}"
            )
            elapsed = time.time() - started

            result = {
                **row_meta,
                "correct": correct,
                "n_test": n_test,
                "accuracy": acc,
                "fold_mean_component": acc,
                "status": "ok",
                "seconds": elapsed,
                "counterpart_overlap": 0,
            }
            append_row(FOLD_RESULTS_FILE, result, RESULT_COLUMNS)

            pred_rows = []
            for p in preds:
                pred_rows.append({
                    **{k: row_meta[k] for k in ["experiment", "split_type", "direction", "train_variety", "test_variety", "condition", "epochs", "seed", "fold"]},
                    **p,
                })
            append_prediction_rows(pred_rows)
            print(f"✅ {key} | {correct}/{n_test} = {acc:.4f}")


for variety in ["pidgin", "camfranglais"]:
    run_within(variety, "native_full", MAIN_EPOCHS, "within_native")
    run_within(variety, "english_gloss", MAIN_EPOCHS, "within_gloss")
    run_within(variety, "premise_only", MAIN_EPOCHS, "within_premise_only")
    run_within(variety, "hypothesis_only", MAIN_EPOCHS, "within_hypothesis_only")

print("✅ Within-variety NLI runs complete/resumed.")


# ---- summary (fold results pooled per seed, then mean and sample SD over the 3 seeds) ----
fr = pd.read_csv(FOLD_RESULTS_FILE)
w = fr[fr["split_type"] == "within_5fold"]
w = w[w["experiment"].isin(["within_native", "within_gloss", "within_premise_only", "within_hypothesis_only"])]
per_seed = w.groupby(["train_variety", "condition", "seed"]).apply(lambda g: g["correct"].sum() / g["n_test"].sum()).rename("acc").reset_index()
summ = per_seed.groupby(["train_variety", "condition"])["acc"].agg(["mean", "std", "count"]).round(4)
print("\nWithin-variety accuracy (mean over seeds, sample SD; count = seeds):")
print(summ.to_string())
print("runs completed:", len(w), "of 120")

save_code("02_C3_within_variety_cv", "Within-variety 5-fold CV (3 seeds, 15 epochs): native, English gloss, premise-only, hypothesis-only")