# 02_C5_counterpart_leak_audit
# AIM: Audit only: full-source/full-target transfer with counterpart leakage (not used in main results)
# saved: 2026-10-05 15:56

# C5: AUDIT ONLY - old full-source/full-target transfer (train on all 114 of one variety, test on all 114 of the other).
# The counterpart sentences are shared, so this is a deliberate leak; it is NOT used in the main results. 6 runs, about 6 min.
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
datasets.disable_progress_bar()

OLD_LEAK_COLUMNS = RESULT_COLUMNS


def run_old_transfer(direction, epochs=MAIN_EPOCHS):
    if direction == "pidgin_to_camfranglais":
        source_variety, target_variety = "pidgin", "camfranglais"
        source_df, target_df = pidgin_df, camf_df
    else:
        source_variety, target_variety = "camfranglais", "pidgin"
        source_df, target_df = camf_df, pidgin_df

    # This is intentionally the old protocol: all source items vs all target items.
    shared_item_ids = set(source_df["id"].str.replace("camf_", "item_", regex=False).str.replace("wes_", "item_", regex=False))
    target_item_ids = set(target_df["id"].str.replace("camf_", "item_", regex=False).str.replace("wes_", "item_", regex=False))
    assert len(shared_item_ids & target_item_ids) == 114

    existing = existing_result_keys()
    for seed in MODEL_SEEDS:
        meta = {
            "experiment": "transfer_old_counterpart_leak",
            "split_type": "parallel_content_counterpart_leak",
            "direction": direction,
            "train_variety": source_variety,
            "test_variety": target_variety,
            "condition": "native_full",
            "epochs": epochs,
            "seed": seed,
            "fold": -1,
        }
        key = result_key(meta)
        if key in existing:
            print("SKIP completed:", key)
            continue

        started = time.time()
        correct, n_test, acc, preds = run_training_with_retry(
            source_df.copy().reset_index(drop=True),
            target_df.copy().reset_index(drop=True),
            "native_full", seed, epochs,
            f"OLD_LEAK_{direction}_e{epochs}_s{seed}"
        )
        elapsed = time.time() - started
        result = {
            **meta,
            "correct": correct,
            "n_test": n_test,
            "accuracy": acc,
            "fold_mean_component": np.nan,
            "status": "intentional_counterpart_leak_audit",
            "seconds": elapsed,
            "counterpart_overlap": 114,
        }
        append_row(FOLD_RESULTS_FILE, result, OLD_LEAK_COLUMNS)

        pred_rows = []
        for p in preds:
            pred_rows.append({
                **{k: meta[k] for k in ["experiment", "split_type", "direction", "train_variety", "test_variety", "condition", "epochs", "seed", "fold"]},
                **p,
            })
        append_prediction_rows(pred_rows)
        print(f"⚠️ OLD LEAK AUDIT {key} | {correct}/{n_test} = {acc:.4f}")


run_old_transfer("pidgin_to_camfranglais", MAIN_EPOCHS)
run_old_transfer("camfranglais_to_pidgin", MAIN_EPOCHS)

print("✅ Old counterpart-leak audit complete. These values are NOT used in Table 1.")


fr = pd.read_csv(FOLD_RESULTS_FILE)
a = fr[fr["experiment"] == "transfer_old_counterpart_leak"]
print("\nCounterpart-leak audit accuracy (per seed, then mean, sample SD):")
print(a.groupby("direction")["accuracy"].agg(["mean", "std", "count"]).round(4).to_string())
print("runs completed:", len(a), "of 6")

save_code("02_C5_counterpart_leak_audit", "Audit only: full-source/full-target transfer with counterpart leakage (not used in main results)")