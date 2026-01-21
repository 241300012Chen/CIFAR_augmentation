import os
import re
import pandas as pd

RESULTS_DIR = r"CIFAR_augmentation\results"  
FINAL_EPOCH = 40
OUT_TEX = "appendix_per_seed_table.tex"

pattern = re.compile(r"^(baseline|geometric|photometric|composite)_frac(10|20)_seed([0-2])\.csv$")

method_name = {
    "baseline": "Baseline",
    "geometric": "Geometric",
    "photometric": "Photometric",
    "composite": "Composite",
}
method_order = {"Baseline": 0, "Geometric": 1, "Photometric": 2, "Composite": 3}

rows = []

for fn in sorted(os.listdir(RESULTS_DIR)):
    m = pattern.match(fn)
    if not m:
        continue

    method, frac, seed = m.group(1), int(m.group(2)), int(m.group(3))
    df = pd.read_csv(os.path.join(RESULTS_DIR, fn))

    if (df["Epoch"] == FINAL_EPOCH).any():
        last = df[df["Epoch"] == FINAL_EPOCH].iloc[-1]
    else:
        last = df.iloc[-1]

    test_acc = float(last["Test_Acc"]) * 100.0
    gap = float(last["Gap"]) * 100.0

    rows.append((frac, method_name[method], seed, test_acc, gap))

rows.sort(key=lambda x: (x[0], method_order[x[1]], x[2]))

with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write("% Auto-generated from results/*.csv\n")
    for frac, mname, seed, acc, gap in rows:
        f.write(f"{frac}\\% & {mname} & {seed} & {acc:.2f} & {gap:.2f} \\\\\n")

print(f"Written {OUT_TEX} with {len(rows)} rows.")
