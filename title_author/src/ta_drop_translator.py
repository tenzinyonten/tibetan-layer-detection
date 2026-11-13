"""Drop works whose gold author span contains ལོ་ཙཱ or ལོ་ཙྪ (translator word). Writes new copies; the inputs are not changed."""
import re, shutil, json
from pathlib import Path
import pandas as pd
from datasets import load_from_disk, DatasetDict

P = Path(__file__).resolve().parents[1] / "data/processed"
SRC_HF, SRC_PQ = P / "gold_dataset_all_strat_single_v3_both", P / "parquet_export_strat_both"
DST_HF, DST_PQ = P / "gold_dataset_all_strat_single_v3_both_notrans", P / "parquet_export_strat_both_notrans"
PAT = re.compile(r"ལོ་ཙཱ|ལོ་ཙྪ|ལོཙཱ|ལོཙྪ|ལོ་ཙ་བ")
bad = lambda a: bool(PAT.search(a or ""))

DST_PQ.mkdir(exist_ok=True); stats = {}; dropped = []
for s in ("train", "validation", "test"):
    df = pd.read_parquet(SRC_PQ / f"{s}.parquet"); m = df.author.map(bad)
    dropped += [(s, w, a) for w, a in zip(df.work_id[m], df.author[m])]
    df[~m].to_parquet(DST_PQ / f"{s}.parquet", index=False); stats[s] = [len(df), int(m.sum()), int((~m).sum())]
ids = {w for _, w, _ in dropped}
ds = load_from_disk(str(SRC_HF)); print(ds["train"].column_names)
out = DatasetDict({s: ds[s].filter(lambda r: r["work_id"] not in ids) for s in ds})
out.save_to_disk(str(DST_HF))
pd.DataFrame(dropped, columns=["split", "work_id", "author"]).to_csv(P / "translator_dropped.csv", index=False)
for f in ("works_index.csv", "README.md"):
    if (SRC_PQ / f).exists(): shutil.copy(SRC_PQ / f, DST_PQ / f)
print(json.dumps(stats), {s: len(out[s]) for s in out})
