# 02_C4_transfer_disjoint
# AIM: MAIN cross-variety transfer with disjoint items (native and hypothesis-only, both directions, 3 seeds x 5 folds)
# saved: 2026-10-05 14:29

# C4: MAIN cross-variety transfer with disjoint items (fold k: source items outside k -> train; target items in k -> test).
# Native text and hypothesis-only baseline, both directions, 3 seeds x 5 folds = 60 runs (about 55 min). Resumable.
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
datasets.disable_progress_bar()

def run_disjoint_transfer(direction, condition="native_full", epochs=MAIN_EPOCHS):
    if direction == "pidgin_to_camfranglais":
        source_variety, target_variety = "pidgin", "camfranglais"
        source_df, target_df = pidgin_df, camf_df
    elif direction == "camfranglais_to_pidgin":
        source_variety, target_variety = "camfranglais", "pidgin"
        source_df, target_df = camf_df, pidgin_df
    else:
        raise ValueError(direction)

    existing = existing_result_keys()

    for seed in MODEL_SEEDS:
        for fold_id in range(N_FOLDS):
            source_train = source_df[source_df["fold"] != fold_id].copy().reset_index(drop=True)
            target_test = target_df[target_df["fold"] == fold_id].copy().reset_index(drop=True)

            source_train_items = set(source_train["id"].str.replace("camf_", "item_", regex=False).str.replace("wes_", "item_", regex=False))
            target_test_items = set(target_test["id"].str.replace("camf_", "item_", regex=False).str.replace("wes_", "item_", regex=False))
            overlap = source_train_items & target_test_items
            assert not overlap, f"COUNTERPART LEAKAGE detected: {sorted(overlap)[:5]}"
            assert len(source_train) == 114 - len(target_test)
            assert len(target_test) in {22, 23}

            meta = {
                "experiment": "transfer_disjoint",
                "split_type": "disjoint_item_5fold",
                "direction": direction,
                "train_variety": source_variety,
                "test_variety": target_variety,
                "condition": condition,
                "epochs": epochs,
                "seed": seed,
                "fold": fold_id,
            }
            key = result_key(meta)
            if key in existing:
                print("SKIP completed:", key)
                continue

            started = time.time()
            correct, n_test, acc, preds = run_training_with_retry(
                source_train, target_test, condition, seed, epochs,
                f"transfer_{direction}_{condition}_e{epochs}_s{seed}_f{fold_id}"
            )
            elapsed = time.time() - started

            result = {
                **meta,
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
                    **{k: meta[k] for k in ["experiment", "split_type", "direction", "train_variety", "test_variety", "condition", "epochs", "seed", "fold"]},
                    **p,
                })
            append_prediction_rows(pred_rows)
            print(f"✅ {key} | {correct}/{n_test} = {acc:.4f}")


run_disjoint_transfer("pidgin_to_camfranglais", "native_full", MAIN_EPOCHS)
run_disjoint_transfer("camfranglais_to_pidgin", "native_full", MAIN_EPOCHS)
run_disjoint_transfer("pidgin_to_camfranglais", "hypothesis_only", MAIN_EPOCHS)
run_disjoint_transfer("camfranglais_to_pidgin", "hypothesis_only", MAIN_EPOCHS)

print("✅ Disjoint-item transfer runs complete/resumed.")


# ---- summary (fold results pooled per seed, then mean and sample SD over the 3 seeds) ----
fr = pd.read_csv(FOLD_RESULTS_FILE)
t = fr[fr["experiment"] == "transfer_disjoint"]
per_seed = t.groupby(["direction", "condition", "seed"]).apply(lambda g: g["correct"].sum() / g["n_test"].sum()).rename("acc").reset_index()
summ = per_seed.groupby(["direction", "condition"])["acc"].agg(["mean", "std", "count"]).round(4)
print("\nDisjoint-item transfer accuracy (mean over seeds, sample SD; count = seeds):")
print(summ.to_string())
print("runs completed:", len(t), "of 60")

save_code("02_C4_transfer_disjoint", "MAIN cross-variety transfer with disjoint items (native and hypothesis-only, both directions, 3 seeds x 5 folds)")