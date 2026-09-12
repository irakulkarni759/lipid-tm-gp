"""Precompute predictions for the web page.

A static page cannot run the GP, so we evaluate it ahead of time over every
binary pair of the supported lipids on a 5% grid and ship the result as JSON.
"""
import os
import sys
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from predict import predict_tm

TIER1 = ["DPPC", "DMPC", "DOPC", "DSPC", "POPC", "Cholesterol", "DMPG", "PSM", "SSM"]
TIER2 = ["DLPC", "DLPE", "DMPE", "DPPE", "DSPE", "DOPE", "POPE",
         "DLPG", "DPPG", "DSPG", "DOPG", "POPG",
         "DLPS", "DMPS", "DPPS", "DSPS", "DOPS", "POPS",
         "DLPA", "DMPA", "DPPA", "DSPA", "DOPA", "POPA"]
STEP = 5

desc = pd.read_csv(os.path.join(ROOT, "data", "lipid_descriptors.csv"))
smiles = dict(zip(desc.lipid, desc.smiles))
names = [n for n in TIER1 + TIER2 if n in smiles]
tier = {n: (1 if n in TIER1 else 2) for n in names}

long = pd.read_csv(os.path.join(ROOT, "data", "measurements_long_qc.csv"))
t = long[long.has_tm & long.meas_smiles_complete & long.meas_identity_ok]
counts = t.groupby("lipid").size().to_dict()

grid = list(range(0, 101, STEP))
out = {"lipids": [{"name": n, "tier": tier[n], "n": int(counts.get(n, 0))}
                  for n in names],
       "step": STEP, "pairs": {}}

for i, a in enumerate(names):
    for b in names[i:]:
        key = f"{a}|{b}"
        row = []
        for pct_b in grid:
            comps = []
            if pct_b < 100:
                comps.append((smiles[a], (100 - pct_b) / 100))
            if pct_b > 0:
                comps.append((smiles[b], pct_b / 100))
            dsc, s1 = predict_tm(comps, family="calorimetry")
            plate, _ = predict_tm(comps, family="laurdan_plate")
            row.append([round(dsc, 1), round(s1, 1), round(plate, 1)])
        out["pairs"][key] = row
    print(f"  {i+1}/{len(names)}  {a}", flush=True)

dst = os.path.join(ROOT, "docs", "predictions.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"\nwrote docs/predictions.json  "
      f"({len(out['pairs'])} pairs, {os.path.getsize(dst)/1024:.0f} KB)")
