# 02_C7_nli_statistics
# AIM: NLI statistics: per-seed accuracy, one-sided exact binomial tests vs 1/3, Table 1, baselines, appendix and epoch tables
# saved: 2026-10-05 17:13

# C7: NLI statistics and output tables (no training). Per-seed accuracy, exact one-sided binomial test vs 1/3,
# Table 1, baselines, appendix tables and epoch sensitivity (5/10/15/20) -> results/*.csv on Drive.

fold_results = read_csv_or_empty(FOLD_RESULTS_FILE, RESULT_COLUMNS)
assert not fold_results.empty, "No NLI results were produced."

# -----------------------------
# CV / disjoint-transfer seed summaries
# -----------------------------
cv = fold_results[fold_results["fold"] >= 0].copy()
cv["accuracy"] = cv["accuracy"].astype(float)
cv["correct"] = cv["correct"].astype(int)
cv["n_test"] = cv["n_test"].astype(int)

seed_rows = []
for keys, g in cv.groupby([
    "experiment", "split_type", "direction", "train_variety", "test_variety",
    "condition", "epochs", "seed"
], dropna=False):
    exp, split_type, direction, train_v, test_v, condition, epochs, seed = keys
    fold_accs = g["accuracy"].to_numpy(dtype=float)
    pooled_correct = int(g["correct"].sum())
    pooled_n = int(g["n_test"].sum())
    pooled_acc = pooled_correct / pooled_n
    fold_sd = float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else np.nan
    p = float(binomtest(pooled_correct, pooled_n, p=CHANCE_P, alternative="greater").pvalue)
    seed_rows.append({
        "experiment": exp,
        "split_type": split_type,
        "direction": direction,
        "train_variety": train_v,
        "test_variety": test_v,
        "condition": condition,
        "epochs": int(epochs),
        "seed": int(seed),
        "fold_mean_accuracy": float(np.mean(fold_accs)),
        "fold_sample_sd": fold_sd,
        "pooled_correct": pooled_correct,
        "pooled_n": pooled_n,
        "pooled_accuracy": pooled_acc,
        "binomial_p_one_sided": p,
        "binomial_p_lt_0_05": bool(p < 0.05),
    })

seed_level = pd.DataFrame(seed_rows)

# -----------------------------
# Old counterpart-leak audit: no folds, but still compute pooled accuracy and binomial test.
# -----------------------------
old_transfer = fold_results[fold_results["experiment"] == "transfer_old_counterpart_leak"].copy()
old_seed_rows = []
for _, r in old_transfer.iterrows():
    k = int(r["correct"])
    n = int(r["n_test"])
    p = float(binomtest(k, n, p=CHANCE_P, alternative="greater").pvalue)
    old_seed_rows.append({
        "experiment": r["experiment"],
        "split_type": r["split_type"],
        "direction": r["direction"],
        "train_variety": r["train_variety"],
        "test_variety": r["test_variety"],
        "condition": r["condition"],
        "epochs": int(r["epochs"]),
        "seed": int(r["seed"]),
        "fold_mean_accuracy": np.nan,
        "fold_sample_sd": np.nan,
        "pooled_correct": k,
        "pooled_n": n,
        "pooled_accuracy": float(r["accuracy"]),
        "binomial_p_one_sided": p,
        "binomial_p_lt_0_05": bool(p < 0.05),
        "counterpart_overlap": int(r["counterpart_overlap"]),
    })
old_seed_level = pd.DataFrame(old_seed_rows)

# Full per-seed audit includes all CV/transfer conditions plus old audit.
per_seed_all = pd.concat([seed_level, old_seed_level], ignore_index=True, sort=False)
per_seed_all.to_csv(RESULTS_DIR / "per_seed.csv", index=False)

if not old_transfer.empty:
    old_transfer.to_csv(RESULTS_DIR / "appendix_d_old_counterpart_leak.csv", index=False)


def seed_summary_table(df):
    rows = []
    group_cols = [
        "experiment", "split_type", "direction", "train_variety", "test_variety",
        "condition", "epochs"
    ]
    for keys, g in df.groupby(group_cols, dropna=False):
        vals = g["pooled_accuracy"].to_numpy(dtype=float)
        rows.append({
            **dict(zip(group_cols, keys)),
            "n_seeds": len(vals),
            "mean_accuracy": float(np.mean(vals)),
            "sample_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "population_sd": float(np.std(vals, ddof=0)) if len(vals) > 1 else np.nan,
            "seeds_p_lt_0_05": int((g["binomial_p_lt_0_05"] == True).sum()),
        })
    return pd.DataFrame(rows)

summary = seed_summary_table(per_seed_all)
summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

# -----------------------------
# Appendix B: four main within-variety conditions.
# -----------------------------
appendix_b = seed_level[
    (seed_level["experiment"].isin(["within_native", "within_gloss"])) &
    (seed_level["epochs"] == MAIN_EPOCHS)
].copy()
appendix_b.to_csv(RESULTS_DIR / "appendix_b_per_seed_main_conditions.csv", index=False)

# -----------------------------
# Baselines: premise-only / hypothesis-only within-variety + hypothesis-only transfer.
# -----------------------------
base_within = seed_level[
    (seed_level["experiment"].isin(["within_premise_only", "within_hypothesis_only"])) &
    (seed_level["epochs"] == MAIN_EPOCHS)
].copy()
base_hyp_transfer = seed_level[
    (seed_level["experiment"] == "transfer_disjoint") &
    (seed_level["condition"] == "hypothesis_only") &
    (seed_level["epochs"] == MAIN_EPOCHS)
].copy()
baselines = pd.concat([base_within, base_hyp_transfer], ignore_index=True, sort=False)
baselines.to_csv(RESULTS_DIR / "baselines.csv", index=False)
base_within.to_csv(RESULTS_DIR / "baselines_within.csv", index=False)
base_hyp_transfer.to_csv(RESULTS_DIR / "baselines_hypothesis_only_transfer.csv", index=False)

# -----------------------------
# Appendix D: corrected disjoint transfer + old audit.
# -----------------------------
appendix_d_transfer = seed_level[
    (seed_level["experiment"] == "transfer_disjoint") &
    (seed_level["condition"] == "native_full") &
    (seed_level["epochs"] == MAIN_EPOCHS)
].copy()
appendix_d_transfer.to_csv(RESULTS_DIR / "appendix_d_disjoint_transfer_per_seed.csv", index=False)

# Every seed-level CV/transfer result has an exact one-sided binomial test.
per_seed_all.to_csv(RESULTS_DIR / "binomial_tests_all_seed_results.csv", index=False)

# -----------------------------
# Main Table 1: mean +/- SAMPLE SD across pooled seed accuracies.
# -----------------------------
def fmt_pct(mean, sd):
    if pd.isna(mean) or pd.isna(sd):
        return "NA"
    return f"{100*mean:.1f}% ± {100*sd:.1f}%"

main_summary = summary[(summary["epochs"] == MAIN_EPOCHS)].copy()


def get_main(exp, train_v, test_v, condition="native_full"):
    r = main_summary[
        (main_summary["experiment"] == exp) &
        (main_summary["train_variety"] == train_v) &
        (main_summary["test_variety"] == test_v) &
        (main_summary["condition"] == condition)
    ]
    assert len(r) == 1, f"Expected one summary row for {exp}/{train_v}/{test_v}/{condition}, found {len(r)}"
    return fmt_pct(float(r.iloc[0]["mean_accuracy"]), float(r.iloc[0]["sample_sd"]))


table1 = pd.DataFrame([
    {
        "Train data": "Cameroon Pidgin only",
        "Test: Cameroon Pidgin": get_main("within_native", "pidgin", "pidgin"),
        "Test: Camfranglais": get_main("transfer_disjoint", "pidgin", "camfranglais", "native_full"),
    },
    {
        "Train data": "Camfranglais only",
        "Test: Cameroon Pidgin": get_main("transfer_disjoint", "camfranglais", "pidgin", "native_full"),
        "Test: Camfranglais": get_main("within_native", "camfranglais", "camfranglais"),
    },
    {
        "Train data": "English glosses (reference)",
        "Test: Cameroon Pidgin": get_main("within_gloss", "pidgin", "pidgin", "english_gloss"),
        "Test: Camfranglais": get_main("within_gloss", "camfranglais", "camfranglais", "english_gloss"),
    },
    {
        "Train data": "Random baseline (3-class)",
        "Test: Cameroon Pidgin": "33.3%",
        "Test: Camfranglais": "33.3%",
    },
])
table1.to_csv(RESULTS_DIR / "table1_main_nli_matrix.csv", index=False)

# -----------------------------
# Epoch sensitivity: all settings reported equally.
# The 15-epoch rows are reused from the main native run; 5, 10 and 20 are extra runs.
# -----------------------------
epoch_rows = seed_level[
    (seed_level["experiment"].isin(["epoch_sensitivity", "within_native"])) &
    (seed_level["condition"] == "native_full") &
    (seed_level["epochs"].isin(EPOCH_SENSITIVITY))
].copy()
epoch_rows["experiment"] = "epoch_sensitivity"
epoch_rows.to_csv(RESULTS_DIR / "epoch_sensitivity_per_seed.csv", index=False)

epoch_summary = seed_summary_table(epoch_rows)
epoch_summary.to_csv(RESULTS_DIR / "epoch_sensitivity.csv", index=False)

# -----------------------------
# Print main results / both SD conventions.
# -----------------------------
print("\n===== MAIN NLI SUMMARY =====")
print(table1.to_string(index=False))
print("\nSample vs population SD (seed-level pooled accuracy):")
print(main_summary[[
    "experiment", "train_variety", "test_variety", "condition",
    "mean_accuracy", "sample_sd", "population_sd", "seeds_p_lt_0_05"
]].to_string(index=False))
print("\n✅ NLI summaries written to Drive.")

save_code("02_C7_nli_statistics", "NLI statistics: per-seed accuracy, one-sided exact binomial tests vs 1/3, Table 1, baselines, appendix and epoch tables")