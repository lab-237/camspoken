# 01_A3a_repair_csv_commas
# AIM: One-off data prep: re-split rows broken by unquoted commas and fix stray '.,' (original kept in inputs/_backup_before_repair)
# saved: 2026-10-05 10:53

import csv, re, shutil
from pathlib import Path

COLS = ["id", "premise_text", "premise_text_en", "premise_audio",
        "hypothesis_text", "hypothesis_text_en", "label"]
TEXT_COLS = [1, 2, 4, 5]

def repair_csv(path):
    path = Path(path)
    backup_dir = path.parent / "_backup_before_repair"
    backup_dir.mkdir(exist_ok=True)
    if not (backup_dir / path.name).exists():
        shutil.copy2(path, backup_dir / path.name)

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    assert [c.strip() for c in next(csv.reader([lines[0]]))] == COLS, "header is not the expected 7 columns"

    rows, repaired, stray, manual = [COLS], [], [], []
    for n, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        if len(row) != 7:
            parts = re.split(r",(?!\s)", line.strip().strip('"'))   # field borders = commas NOT followed by a space
            if len(parts) != 7:
                manual.append((n, len(parts), line))
                continue
            row = [p.strip() for p in parts]
            repaired.append((n, row[0]))
        for j in TEXT_COLS:                                          # "Today., I" -> "Today, I"
            new = re.sub(r"\.,(\s)", r",\1", row[j])
            if new != row[j]:
                stray.append((row[0], row[j], new))
                row[j] = new
        rows.append(row)
    return rows, repaired, stray, manual

def repair_and_save(path):
    rows, repaired, stray, manual = repair_csv(path)
    print(f"\n{Path(path).name}")
    print(f"  rows re-split (unquoted commas): {len(repaired)} -> {[r[1] for r in repaired]}")
    print(f"  stray full stop before a comma fixed: {len(stray)}")
    for rid, old, new in stray:
        print(f"     {rid}: {old}\n        -> {new}")
    if manual:
        print("  ❗ could not split automatically (file NOT changed):")
        for n, k, line in manual:
            print(f"     line {n} ({k} parts): {line}")
        return False
    assert len(rows) == 115, f"expected 115 lines incl. header, got {len(rows)}"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f, quoting=csv.QUOTE_MINIMAL).writerows(rows)
    print("  ✅ saved with proper quoting. Original kept in inputs/_backup_before_repair/")
    return True

for p in [PIDGIN_CSV, CAMFRANGLAIS_CSV]:
    repair_and_save(p)
