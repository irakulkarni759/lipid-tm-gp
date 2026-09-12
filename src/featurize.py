"""Turn each lipid's SMILES into physical descriptors, then aggregate a mixture
into a single feature vector by mole-fraction weighting.

Two kinds of mixture feature are produced for every descriptor d:
  mean_d  = sum_i x_i * d_i          (ideal-mixing value)
  spread_d= sum_i x_i * |d_i - mean| (how mismatched the components are)

The spread terms matter because binary phase behaviour deviates from a straight
line exactly when the components differ in chain length. A pure weighted mean
can only ever produce a linear Tm vs composition line.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.join(ROOT, "src"))
from chem import describe, DESC_COLS

# QC needs descriptors and descriptors need the parsed table, so the first pass
# runs before QC exists and the second picks up the QC'd file.
_QC = os.path.join(ROOT, "data", "measurements_long_qc.csv")
LONG = _QC if os.path.exists(_QC) else os.path.join(ROOT, "data", "measurements_long.csv")
OUT_LIPID = os.path.join(ROOT, "data") + "/lipid_descriptors.csv"
OUT_MIX = os.path.join(ROOT, "data") + "/mixture_features.csv"



def main():
    long = pd.read_csv(LONG)

    # ---- per-lipid descriptors -------------------------------------------
    uniq = long.dropna(subset=["smiles"])[["lipid", "smiles"]].drop_duplicates("lipid")
    rows, failed = [], []
    for lipid, smi in uniq.itertuples(index=False):
        d = describe(smi)
        if d is None:
            failed.append(lipid)
            continue
        rows.append(dict(lipid=lipid, smiles=smi, **d))
    lip = pd.DataFrame(rows)
    lip.to_csv(OUT_LIPID, index=False)
    print(f"lipids with descriptors: {len(lip)}  (RDKit parse failures: {len(failed)})")
    if failed:
        print("  failed:", failed[:10])

    # ---- aggregate to mixtures -------------------------------------------
    df = long.merge(lip[["lipid"] + DESC_COLS], on="lipid", how="left")
    ok = (df.meas_identity_ok if "meas_identity_ok" in df.columns
          else pd.Series(True, index=df.index))
    usable = df[df.has_tm & df.meas_smiles_complete & ok].copy()
    usable = usable[usable[DESC_COLS].notna().all(axis=1)]

    # Counterion travels with the measurement, not the molecule, so it is
    # attached here rather than in the per-lipid descriptor table.
    ION_COLS = ["ion_valency", "ion_divalent"]
    for c in ION_COLS:
        if c not in usable.columns:
            usable[c] = 0

    recs = []
    for mid, g in usable.groupby("meas_id"):
        x = g["mole_frac"].to_numpy(float)
        x = x / x.sum()
        D = g[DESC_COLS].to_numpy(float)
        mean = x @ D
        spread = x @ np.abs(D - mean)
        rec = {"meas_id": mid}
        rec.update({f"mean_{c}": v for c, v in zip(DESC_COLS, mean)})
        rec.update({f"spread_{c}": v for c, v in zip(DESC_COLS, spread)})
        for c in ION_COLS:
            rec[f"mean_{c}"] = float(np.average(g[c].to_numpy(float), weights=x))
        rec["n_components"] = len(g)
        rec["tm_c"] = g["tm_c"].iloc[0]
        rec["method"] = g["method"].iloc[0]
        rec["dataset"] = g["dataset"].iloc[0]
        rec["reliable"] = bool(g["reliable"].iloc[0])
        rec["system"] = "+".join(sorted(g["lipid"]))
        rec["counterion"] = (g.loc[g["counterion"] != "none", "counterion"].iloc[0]
                             if "counterion" in g.columns
                             and (g["counterion"] != "none").any() else "none")
        rec["paper"] = g["paper"].iloc[0]
        rec["composition"] = " / ".join(
            f"{l} {p:.3g}%" for l, p in zip(g["lipid"], g["mole_frac"] * 100))
        recs.append(rec)

    mix = pd.DataFrame(recs)
    mix.to_csv(OUT_MIX, index=False)
    print(f"mixture feature rows: {len(mix)}   features: "
          f"{sum(c.startswith(('mean_','spread_')) for c in mix.columns)}")
    print(f"unique systems: {mix.system.nunique()}")
    print(f"multi-component: {(mix.n_components>1).sum()}")
    print("\nDMPC/DPPC rows in the final table:")
    dd = mix[mix.system.isin(["DMPC+DPPC", "DMPC", "DPPC"])]
    print(dd[["meas_id", "composition", "tm_c", "method", "reliable"]].to_string(index=False))


if __name__ == "__main__":
    main()
