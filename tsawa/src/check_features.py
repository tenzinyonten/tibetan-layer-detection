#!/usr/bin/env python3
"""
check_features.py — did f4/f5 (source-marker-before, closer-ahead) actually
fire where they should?

The hypothesis for why features didn't help: they were computed wrong, so the
model got noise instead of the particle-boundary signal.

Direct test: for each tsawa span, the LAST token should very often have f5=1
(a closer like ཞེས་/ཅེས་ starts within the next 2 clauses), because that's
exactly the convention — a quote ends, then the particle. If f5 is ~0 on span-
end tokens, the feature is broken.

Reads the built dataset's stored "features" column against the labels. No GPU.

    python check_features.py --dataset ds_v6
"""
import argparse
import numpy as np
from datasets import load_from_disk

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="ds_v6")
ap.add_argument("--split", default="train")
args = ap.parse_args()

ds = load_from_disk(args.dataset)[args.split]
if "features" not in ds.column_names:
    print("No 'features' column — this dataset was built without features.")
    print("Columns:", ds.column_names)
    raise SystemExit

# feature index meaning (from the build spec):
# 0 f1 clause-syllables/20   1 f2 isometric   2 f3 std   3 f4 source-before   4 f5 closer-ahead
names = ["f1_syl", "f2_iso", "f3_std", "f4_src_before", "f5_closer_ahead"]

# accumulate feature means on: span-END tokens, span-INSIDE tokens, O tokens
end_sum = np.zeros(5); end_n = 0
in_sum = np.zeros(5); in_n = 0
o_sum = np.zeros(5); o_n = 0

for row in ds:
    lab = np.array(row["labels"])
    feat = np.array(row["features"], dtype=float)  # [T,5]
    m = lab != -100
    lab = lab[m]; feat = feat[m]
    if len(lab) == 0:
        continue
    # span end = a B/I token whose next token is not I (2)
    is_pos = np.isin(lab, (1, 2))
    for i in range(len(lab)):
        if lab[i] == 0:
            o_sum += feat[i]; o_n += 1
        else:
            nxt = lab[i + 1] if i + 1 < len(lab) else -1
            if nxt != 2:  # span ends here
                end_sum += feat[i]; end_n += 1
            else:
                in_sum += feat[i]; in_n += 1

print(f"tokens — span-end: {end_n:,}  span-inside: {in_n:,}  O: {o_n:,}\n")
print(f"{'feature':<18}{'span-END':>10}{'span-inside':>13}{'O':>10}")
for j, nm in enumerate(names):
    e = end_sum[j]/max(end_n,1); ins = in_sum[j]/max(in_n,1); o = o_sum[j]/max(o_n,1)
    print(f"{nm:<18}{e:>10.3f}{ins:>13.3f}{o:>10.3f}")

print("""
READING:
- f5_closer_ahead should be HIGH on span-END tokens (a closer follows the
  quote) and low on O tokens. If it's ~equal everywhere, f5 is broken.
- f4_src_before should be higher on span-START-ish tokens; here we only split
  end/inside/O, so mainly check it's not all zeros.
- f2_iso should be higher on span tokens (end+inside) than O — verse is
  metrical. If not, the whole feature pipeline is suspect.
If the features look flat/wrong, that explains the null ablation — worth
recomputing before trying CRF.
""")