import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, sys
sys.path.insert(0, os.path.join(ROOT, "src"))
import qc

base = os.path.join(ROOT, "data") + "/"
long = pd.read_csv(base + "measurements_long.csv")
lip = pd.read_csv(base + "lipid_descriptors.csv")
out = qc.run(long, lip)
out.to_csv(base + "measurements_long_qc.csv", index=False)

n_meas = out.meas_id.nunique()
bad = out[out.identity_suspect]
print(f"measurements: {n_meas}")
print(f"component-rows flagged: {len(bad)}")
print(f"measurements flagged:   {n_meas - out.meas_identity_ok.sum() if False else (~out.groupby('meas_id')['meas_identity_ok'].first()).sum()}")
print("\nreason breakdown:")
print(bad.identity_reason.str.split(' !=').str[0].str.slice(0, 40).value_counts().head())
print("\nexamples:")
print(bad[["meas_id", "lipid", "tm_c", "marsh", "identity_reason"]].head(12).to_string(index=False))

clean = out[out.meas_identity_ok & out.has_tm & out.meas_smiles_complete]
print(f"\nclean measurements retained: {clean.meas_id.nunique()}")
dp = clean[(clean.lipid == "DPPC") & (clean.n_components == 1)]
print("DPPC pure Tm after QC:", sorted(dp.tm_c.unique()))
dm = clean[(clean.lipid == "DMPC") & (clean.n_components == 1)]
print("DMPC pure Tm after QC:", sorted(dm.tm_c.unique()))
