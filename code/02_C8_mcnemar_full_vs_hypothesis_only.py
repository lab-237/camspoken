# 02_C8_mcnemar_full_vs_hypothesis_only
# AIM: Paired exact McNemar test, full input vs hypothesis-only on the same items, per seed (12 tests)
# saved: 2026-10-05 20:52

# C8: paired exact McNemar test - full input (native text) vs hypothesis-only, on the SAME test items, one test per seed.
# Question answered: is the full-input model better than a model that sees only the hypothesis? (The binomial test in C7
# answers a different question: is a single model better than chance?) 12 tests = 2 within-variety + 2 transfer settings x 3 seeds.
from scipy.stats import binomtest
preds = pd.read_csv(PREDICTIONS_FILE)
preds = preds[preds["epochs"] == MAIN_EPOCHS]

def side(exp, cond, tv, sv, seed, direction=None):
    q = preds[(preds["experiment"] == exp) & (preds["condition"] == cond) & (preds["train_variety"] == tv) &
              (preds["test_variety"] == sv) & (preds["seed"] == seed)]
    if direction: q = q[q["direction"] == direction]
    assert len(q) == 114 and q["id"].is_unique, (exp, cond, tv, sv, seed, len(q))
    return q.set_index("id")["correct"].astype(int)

settings = [("within", "within_native", "within_hypothesis_only", "pidgin", "pidgin", None),
            ("within", "within_native", "within_hypothesis_only", "camfranglais", "camfranglais", None),
            ("transfer", "transfer_disjoint", "transfer_disjoint", "pidgin", "camfranglais", "pidgin_to_camfranglais"),
            ("transfer", "transfer_disjoint", "transfer_disjoint", "camfranglais", "pidgin", "camfranglais_to_pidgin")]
rows = []
for kind, e_full, e_hyp, tv, sv, dr in settings:
    for seed in MODEL_SEEDS:
        a = side(e_full, "native_full", tv, sv, seed, dr)
        b = side(e_hyp, "hypothesis_only", tv, sv, seed, dr).reindex(a.index)
        full_only = int(((a == 1) & (b == 0)).sum()); hyp_only = int(((a == 0) & (b == 1)).sum())
        disc = full_only + hyp_only
        p = 1.0 if disc == 0 else float(binomtest(min(full_only, hyp_only), disc, 0.5, alternative="two-sided").pvalue)
        rows.append(dict(setting=kind, train_variety=tv, test_variety=sv, seed=seed, n=len(a),
                         acc_full=a.mean(), acc_hyp_only=b.mean(), diff_full_minus_hyp=a.mean() - b.mean(),
                         full_right_hyp_wrong=full_only, hyp_right_full_wrong=hyp_only, mcnemar_p_two_sided=p, p_lt_0_05=p < 0.05))
mc = pd.DataFrame(rows)
mc.to_csv(RESULTS_DIR / "mcnemar_full_vs_hypothesis_only.csv", index=False)
print(mc.round(3).to_string(index=False))
print(f"\nSeed-level tests with p < 0.05: {int(mc['p_lt_0_05'].sum())} of {len(mc)}")
print("Direction of the difference (full - hypothesis-only > 0 in", int((mc['diff_full_minus_hyp'] > 0).sum()), "of", len(mc), "tests)")

save_code("02_C8_mcnemar_full_vs_hypothesis_only", "Paired exact McNemar test, full input vs hypothesis-only on the same items, per seed (12 tests)")