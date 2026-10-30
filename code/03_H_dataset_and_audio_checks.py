# 03_H_dataset_and_audio_checks
# AIM: Dataset format, text statistics, scenario matching, negation check and audio statistics for Appendix H
# saved: 2026-10-24 12:35

# 03_H_dataset_and_audio_checks
# AIM: Dataset format, text statistics, scenario matching, negation check and audio statistics for Appendix H
# saved: 2026-10-06 02:18

import re, io, unicodedata
from collections import Counter
P = pidgin_df.copy(); C = camf_df.copy()
print("columns:", P.columns.tolist())
raw = (FINAL_CSV_DIR/"pidgin.csv").read_bytes()[:3]; print("BOM:", raw == b"\xef\xbb\xbf")
col = lambda d, *names: next(n for n in names if n in d.columns)
pc, ph = col(P,"premise_text","premise"), col(P,"hypothesis_text","hypothesis")
pe, he = col(P,"premise_text_en","premise_en"), col(P,"hypothesis_text_en","hypothesis_en")
cc, ch = col(C,"premise_text","premise"), col(C,"hypothesis_text","hypothesis")
ce, che = col(C,"premise_text_en","premise_en"), col(C,"hypothesis_text_en","hypothesis_en")
wc = lambda s: len(str(s).split())
def stats(d, a, b, name):
    print(f"\n== {name} ==  rows {len(d)}, empty cells {int(d.isna().sum().sum() + (d.astype(str).apply(lambda s: s.str.strip()=='').sum().sum()))}")
    print(" newline in any field:", bool(d.astype(str).apply(lambda s: s.str.contains('\n')).any().any()),
          "| curly apostrophe:", bool(d.astype(str).apply(lambda s: s.str.contains('[’‘]')).any().any()))
    print(" unique premises/hypotheses/pairs:", d[a].nunique(), d[b].nunique(), len(d[[a,b]].drop_duplicates()))
    for lab, x in ((a,"premise"),(b,"hypothesis")):
        w = d[lab].map(wc); print(f" {x} words total/mean/min-max: {w.sum()} / {w.mean():.1f} / {w.min()}-{w.max()}")
    print(" premises ending . ! ?:", int(d[a].astype(str).str.strip().str[-1].isin(list('.!?')).sum()), "| with '?':", int(d[a].astype(str).str.contains(r'\?').sum()))
    print(" labels:", d.label.value_counts().to_dict())
    print(" mean hypothesis words by label:", d.assign(w=d[b].map(wc)).groupby("label").w.mean().round(1).to_dict())
stats(P, pc, ph, "Pidgin"); stats(C, cc, ch, "Camfranglais")
print("\nGloss premise words (P / C):", P[pe].map(wc).sum(), C[ce].map(wc).sum(), "| gloss hypothesis words:", P[he].map(wc).sum(), C[che].map(wc).sum())
diff = [(i+1, c, P[c].iloc[i], C[c2].iloc[i]) for c, c2 in ((pe,ce),(he,che)) for i in range(114) if str(P[c].iloc[i]).strip()!=str(C[c2].iloc[i]).strip()]
print("gloss cells that differ between files:", len(diff)); [print("  ", d) for d in diff]
print("labels agree:", int((P.label.values == C.label.values).sum()), "of 114")
tok = lambda s: set(re.findall(r"\w+", str(s).lower()))
jac = [len(tok(a)&tok(b))/max(1,len(tok(a)|tok(b))) for a, b in zip(P[pc], C[cc])]
print("mean word overlap (Jaccard) between counterpart premises: %.1f%%" % (100*np.mean(jac)), "| identical pairs:", int(sum(str(a).strip().lower()==str(b).strip().lower() for a, b in zip(P[pc], C[cc]))))
neg_en = {"no","never","nobody","nothing","none","not","neither","nor","nowhere","cannot","dont","doesnt","didnt","wont","cant","isnt","arent","wasnt"}
neg_fr = {"ne","pas","jamais","aucun","aucune","personne","rien","plus","non"}
def has(s, words, fr=False):
    s = str(s).lower().replace("’","'"); t = set(re.findall(r"[a-zà-ÿ]+", s.replace("'", "")))
    return bool(t & words) or (fr and bool(re.search(r"\bn'", s)))
rows = []
for name, d, c, w, fr in (("Pidgin",P,ph,neg_en,False),("Camfranglais",C,ch,neg_fr,True),("English glosses (Pidgin material)",P,he,neg_en,False)):
    r = {lab: f"{int(g[c].map(lambda s: has(s,w,fr)).sum())} of {len(g)}" for lab, g in d.groupby("label")}; rows.append((name, r))
print("\nNegation words in hypotheses:"); [print(" ", n, r) for n, r in rows]
a = pd.read_csv(RESULTS_DIR/"audio_preprocessing_log.csv")
for v, g in a.groupby("variety"):
    print(f"\naudio {v}: final mean {g.final_sec.mean():.2f} sd {g.final_sec.std():.2f} min/median/max {g.final_sec.min():.2f}/{g.final_sec.median():.2f}/{g.final_sec.max():.2f} total {g.final_sec.sum():.0f}s | raw mean {g.source_sec.mean():.2f} | lead removed mean {g.lead_removed_sec.mean():.2f} | tail kept mean {g.tail_kept_sec.mean():.2f}")
save_code("03_H_dataset_and_audio_checks", "Dataset format, text statistics, scenario matching, negation check and audio statistics for Appendix H")