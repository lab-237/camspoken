# 03_D4_zero_error_and_leak_check
# AIM: Zero-error clip counts (Large-v3) and per-seed leak-audit values for Appendix C.7 and D.4
# saved: 2026-10-06 02:09

t = pd.read_csv(RESULTS_DIR/"appendix_c_asr_transcripts.csv", keep_default_na=False)
t = t[t.model=="large-v3"].copy()
t["err"] = t.substitutions + t.deletions + t.insertions
print("Large-v3 clips with zero word errors (of 114):")
print(t.assign(zero=t.err==0).groupby(["variety","condition"]).zero.sum().to_string())
print("\nLeak audit, per seed:")
d = pd.read_csv(RESULTS_DIR/"appendix_d_old_counterpart_leak.csv")
print(d.round(4).to_string(index=False))
save_code("03_D4_zero_error_and_leak_check", "Zero-error clip counts (Large-v3) and per-seed leak-audit values for Appendix C.7 and D.4")