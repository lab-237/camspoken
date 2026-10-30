# 01_A3b_tidy_text
# AIM: One-off data prep: collapse double spaces and add a final full stop where missing
# saved: 2026-10-05 10:53

import pandas as pd, re, csv

TEXT_COLUMNS = ["premise_text", "premise_text_en", "hypothesis_text", "hypothesis_text_en"]
frames = {}

for path in [PIDGIN_CSV, CAMFRANGLAIS_CSV]:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    changes = 0
    for col in TEXT_COLUMNS:
        for i, old in df[col].items():
            new = re.sub(r"\s{2,}", " ", old).strip()          # double spaces -> one space
            if not re.search(r"[.!?]$", new):                    # no final stop -> add one
                new += "."
            if new != old:
                changes += 1
                print(f"{df.at[i, 'id']} [{col}]\n   {old!r}\n-> {new!r}")
                df.at[i, col] = new
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    frames[path.name] = df
    print(f"\n✅ {path.name}: {changes} cell(s) tidied and saved.\n")

p = frames[PIDGIN_CSV.name]; c = frames[CAMFRANGLAIS_CSV.name]
for col in ["premise_text_en", "hypothesis_text_en"]:
    diff = p[col] != c[col]
    print(f"English gloss differences in {col}: {int(diff.sum())}")
    for i in p.index[diff]:
        print(f"  {p.at[i,'id'][-5:]}\n     Pidgin gloss      : {p.at[i,col]}\n     Camfranglais gloss: {c.at[i,col]}")
