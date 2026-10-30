# 01_A3c_standardize_gloss
# AIM: One-off data prep: one identical English gloss for both varieties (8 edits) + report-only scan for other problems
# saved: 2026-10-05 10:53

import pandas as pd, re, csv

# ---- 1. ONE standard English gloss for both varieties (edit any line you disagree with) ----
STANDARD = {
    ("premise_text_en",    "00045"): "The baby is sleeping quietly. Don't get him/her up.",
    ("premise_text_en",    "00049"): "I ate achu (a local dish) at the baby welcome party.",
    ("premise_text_en",    "00059"): "I initially wanted to accept your sale offer.",
    ("premise_text_en",    "00075"): "The bike rider charged me 500 francs to drop me at University Junction.",
    ("premise_text_en",    "00076"): "Today, I have had a new idea on my mind.",
    ("premise_text_en",    "00083"): "I am in disbelief at that girl for the kind of lies she tells.",
    ("premise_text_en",    "00084"): "He locked the door tightly before going to sleep.",
    ("hypothesis_text_en", "00052"): "I am not laughing at all at what you said.",
}

dfs = {}
for path, prefix in [(PIDGIN_CSV, "wes"), (CAMFRANGLAIS_CSV, "camf")]:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for (col, num), text in STANDARD.items():
        i = df.index[df["id"] == f"{prefix}_{num}"][0]
        df.at[i, col] = text
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    dfs[prefix] = df

for col in ["premise_text_en", "hypothesis_text_en"]:
    n = int((dfs["wes"][col].values != dfs["camf"][col].values).sum())
    print(f"English gloss differences left in {col}: {n}")
    assert n == 0, "the two glosses are still not identical"
print("✅ Both CSVs now share ONE identical English gloss for all 114 items.\n")

# ---- 2. Scan for other problems (REPORT ONLY, nothing is changed) ----
CHECKS = [
    ("space before punctuation",     r"\s[.,!?;:]",                    None),
    ("no space after punctuation",   r"[.,!?;:][A-Za-zÀ-ÿ]",           None),
    ("double / odd punctuation",     r"[.!?]{2,}|[!?][.,]|[.,][!?]|,,", None),
    ("slash with spaces",            r"\s/|/\s",                       None),
    ("no space before parenthesis",  r"\w\(",                          None),
    ("curly quotes / special dash",  r"[’‘“”…–—]",                    None),
    ("repeated word (English only)", r"\b(\w+)\s+\1\b",                "gloss"),
    ("gloss starts lowercase",       r"^[a-z]",                        "gloss"),
]
found = 0
for prefix, df in dfs.items():
    for col in ["premise_text", "premise_text_en", "hypothesis_text", "hypothesis_text_en"]:
        is_gloss = col.endswith("_en")
        for name, pat, scope in CHECKS:
            if scope == "gloss" and not is_gloss:
                continue
            for i, s in df[col].items():
                if re.search(pat, s, flags=re.I if "repeated" in name else 0):
                    found += 1
                    print(f"[{prefix} {df.at[i,'id'][-5:]}] {col} | {name}: {s}")
print(f"\nScan finished: {found} item(s) to look at." if found else "\nScan finished: nothing suspicious found.")

# capital vs lowercase starts in the native text (information only)
for prefix, df in dfs.items():
    for col in ["premise_text", "hypothesis_text"]:
        up = int(df[col].str.match(r"^[A-ZÀ-Ý]").sum())
        print(f"{prefix} {col}: {up} start with a capital letter, {114-up} with a lowercase letter")
