# 03_G_fold_table_print
# AIM: Print fold sizes, label counts and item lists for Appendix G
# saved: 2026-10-06 02:14

f = pd.read_csv(RESULTS_DIR/"shared_item_folds.csv")
print(f.columns.tolist()); print(f.head(3).to_string(index=False))
c = [x for x in f.columns if "fold" in x.lower()][0]
n = [x for x in f.columns if "item" in x.lower() or x=="id"][0]
print(pd.crosstab(f[c], pidgin_df.set_index("id").loc[f[n] if f[n].dtype==object else [f"wes_{i:05d}" for i in f[n]], "label"].values.ravel() if False else pidgin_df["label"].values[:len(f)]).to_string())
for k, g in f.groupby(c): print(k, len(g), sorted(g[n].tolist()))
save_code("03_G_fold_table_print", "Print fold sizes, label counts and item lists for Appendix G")