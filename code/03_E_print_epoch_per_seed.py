# 03_E_print_epoch_per_seed
# AIM: Print per-seed epoch-sensitivity results for Appendix E.2
# saved: 2026-10-06 02:11

e = pd.read_csv(RESULTS_DIR/"epoch_sensitivity_per_seed.csv")
cols = [c for c in ["train_variety","epochs","seed","pooled_correct","pooled_n","pooled_accuracy","binomial_p_one_sided"] if c in e.columns]
print(e[cols].round(4).to_string(index=False))
save_code("03_E_print_epoch_per_seed", "Print per-seed epoch-sensitivity results for Appendix E.2")