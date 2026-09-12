"""Add a DSC-equivalent Tm column to the flat dataset.

The GP learns a per-instrument offset. For each measurement we ask the model
what it would predict on that row's own instrument, and what it would predict
on calorimetry, and take the difference. Subtracting that from the measured
value puts every row on one comparable scale.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pickle
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.join(ROOT, "src"))

BASE = ROOT + os.sep
mix = pd.read_csv(BASE + "data/mixture_features.csv")
with open(BASE + "data/gp_model.pkl", "rb") as f:
    m = pickle.load(f)


def family(method):
    s = str(method)
    if s == "Laurdan GP (plate reader)":
        return "laurdan_plate"
    if "Laurdan" in s:
        return "laurdan_other"
    if any(k in s for k in ("DSC", "compilation", "unspecified", "X-ray")):
        return "calorimetry"
    return "other_probe"


mix["family"] = mix["method"].map(family)
Xd = m["scaler"].transform(mix[m["feat_cols"]].to_numpy(float))


def pred(fam_of_row):
    Xm = np.stack([(fam_of_row == f).astype(float) for f in m["fam_levels"]], axis=1)
    return m["gp"].predict(np.hstack([Xd, Xm]))


own = pred(mix["family"].to_numpy())
dsc = pred(np.array(["calorimetry"] * len(mix)))
mix["method_offset_c"] = (own - dsc).round(2)
mix["tm_dsc_equiv_c"] = (mix["tm_c"] - mix["method_offset_c"]).round(2)

cols = ["meas_id", "composition", "system", "n_components", "tm_c",
        "method", "family", "method_offset_c", "tm_dsc_equiv_c",
        "dataset", "reliable"]
feat = [c for c in mix.columns if c.startswith(("mean_", "spread_"))]
out = mix[cols + feat]
out.to_csv(BASE + "data/dataset_final.csv", index=False)

print("mean learned offset by instrument family (deg C vs calorimetry):")
print(mix.groupby("family")["method_offset_c"].agg(["mean", "count"]).round(2).to_string())
print("\nAria's DMPC/DPPC series, raw vs corrected:")
a = mix[mix.dataset == "aria_2026"].sort_values("composition")
print(a[["composition", "tm_c", "method_offset_c", "tm_dsc_equiv_c", "reliable"]].to_string(index=False))
print(f"\nwrote data/dataset_final.csv  ({len(out)} rows, {len(out.columns)} cols)")
