"""Build the two files that matter, from the QC'd data.

    lipids.csv        the dictionary. one row per lipid.
    measurements.csv  the data. one row per measurement.

Scope is mixtures plus the pure-lipid readings for the lipids that appear in
them. One row = one measurement, always. SMILES sit inline.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np

BASE = ROOT + os.sep
MAXC = 3

long = pd.read_csv(BASE + "data/measurements_long_qc.csv")
long = long[long.has_tm & long.meas_smiles_complete & long.meas_identity_ok]

mix_ids = long.loc[long.n_components > 1, "meas_id"].unique()
in_mix = set(long.loc[long.meas_id.isin(mix_ids), "lipid"])
keep = long[long.meas_id.isin(mix_ids) |
            ((long.n_components == 1) & long.lipid.isin(in_mix))]

# ---------------------------------------------------------------- lipids.csv
desc = pd.read_csv(BASE + "data/lipid_descriptors.csv")
lip = (keep[["lipid", "smiles"]].drop_duplicates("lipid")
       .merge(desc, on=["lipid", "smiles"], how="left"))
lip["chains"] = lip.apply(
    lambda r: "none (sterol)" if r.n_chains == 0 else
              (f"{int(r.chain_min)}:{int(r.unsat_total/max(r.n_chains,1))}"
               f" x{int(r.n_chains)}" if r.chain_min == r.chain_max else
               f"{int(r.chain_min)}/{int(r.chain_max)}"), axis=1)
lip["geometry"] = np.where(lip.n_cis > 0, "cis",
                    np.where(lip.n_trans > 0, "trans", "saturated"))
lipids = lip[["lipid", "smiles", "headgroup", "chains", "n_chains",
              "chain_min", "chain_max", "unsat_total", "geometry",
              "mw", "charge"]].rename(columns={
    "lipid": "name", "n_chains": "n_tails", "chain_min": "shortest_tail_c",
    "chain_max": "longest_tail_c", "unsat_total": "n_double_bonds",
    "mw": "molecular_weight"}).sort_values("name")
lipids.to_csv(BASE + "data/lipids.csv", index=False)

# ---------------------------------------------------------- measurements.csv
offs = pd.read_csv(BASE + "data/dataset_final.csv")[
    ["meas_id", "method_offset_c", "tm_dsc_equiv_c"]]

rows = []
for mid, g in keep.groupby("meas_id"):
    g = g.sort_values("mole_frac", ascending=False)
    r = {"meas_id": mid,
         "source": "Aria (experiment)" if g.dataset.iloc[0] == "aria_2026"
                   else "literature",
         "n_lipids": len(g)}
    for i in range(MAXC):
        has = i < len(g)
        r[f"lipid_{i+1}"] = g.lipid.iloc[i] if has else ""
        r[f"mol_pct_{i+1}"] = round(g.mole_frac.iloc[i]*100, 1) if has else np.nan
        r[f"smiles_{i+1}"] = g.smiles.iloc[i] if has else ""
    r["tm_measured_c"] = g.tm_c.iloc[0]
    r["method"] = g.method.iloc[0]
    r["trusted"] = bool(g.reliable.iloc[0])
    r["paper"] = str(g.paper.iloc[0])[:70] if pd.notna(g.paper.iloc[0]) else ""
    r["doi"] = g.doi.iloc[0] if pd.notna(g.doi.iloc[0]) else ""
    rows.append(r)

m = pd.DataFrame(rows).merge(offs, on="meas_id", how="left").rename(
    columns={"method_offset_c": "instrument_offset_c",
             "tm_dsc_equiv_c": "tm_corrected_c"})
cols = ["meas_id", "source", "n_lipids"]
for i in range(1, MAXC+1):
    cols += [f"lipid_{i}", f"mol_pct_{i}", f"smiles_{i}"]
cols += ["tm_measured_c", "instrument_offset_c", "tm_corrected_c",
         "method", "trusted", "paper", "doi"]
m = m[cols].sort_values(["n_lipids", "lipid_1", "mol_pct_1"],
                        ascending=[True, True, False])
m.to_csv(BASE + "data/measurements.csv", index=False)

print(f"lipids.csv        {len(lipids):3d} rows, {len(lipids.columns)} cols")
print(f"measurements.csv  {len(m):3d} rows, {len(m.columns)} cols  "
      f"({(m.n_lipids>1).sum()} mixtures, {(m.n_lipids==1).sum()} pure)")
print("\n--- lipids.csv ---")
print(lipids[["name", "headgroup", "chains", "geometry"]].to_string(index=False))
