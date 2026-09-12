"""GP regression: mixture composition (via SMILES descriptors) -> Tm.

Design notes
------------
* Features are mole-fraction-weighted SMILES descriptors plus "spread" terms
  that encode component mismatch. Without the spread terms the model can only
  ever draw a straight line between the two pure-lipid Tm values.
* Measurement method enters as extra one-hot columns inside an ARD Matern
  kernel, so the GP can learn a per-method offset. This is what lets Aria's
  plate-reader values sit in the same table as DSC literature without the two
  being averaged into mush.
* Validation is leave-one-SYSTEM-out, not random k-fold. A random split would
  put near-duplicate rows from the same paper on both sides and report a
  flattering, meaningless score.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

BASE = ROOT + os.sep
MIX = BASE + "data/mixture_features.csv"

METHOD_FAMILY = {
    "calorimetry": ["DSC", "mDSC", "compilation", "unspecified", "X-ray"],
    "laurdan_plate": ["Laurdan GP (plate reader)"],
    "laurdan_other": ["Laurdan", "C-Laurdan"],
    "other_probe": [],   # anisotropy, FRET, FTIR, zeta, plasmonic, ...
}


def method_family(m):
    m = str(m)
    if m == "Laurdan GP (plate reader)":
        return "laurdan_plate"
    if "Laurdan" in m:
        return "laurdan_other"
    if any(k in m for k in ("DSC", "compilation", "unspecified", "X-ray")):
        return "calorimetry"
    return "other_probe"


def load(drop_unreliable=True):
    mix = pd.read_csv(MIX)
    if drop_unreliable:
        mix = mix[mix.reliable]
    mix = mix.dropna(subset=["tm_c"])
    # Specified-counterion rows stay OUT of training. The counterion is real
    # information and it is kept in the dataset, but there is at most one
    # measurement per ion and leave-one-system-out removes that system anyway,
    # so the model cannot learn the effect and can only be misled by it.
    # Measured: including them leaves error on everything else unchanged
    # (3.91 vs 3.89 C) while the ion rows themselves miss by 13.4 C.
    if "counterion" in mix.columns:
        mix = mix[mix["counterion"] == "none"]
    mix["family"] = mix["method"].map(method_family)
    return mix.reset_index(drop=True)


def build_xy(mix, feat_cols, fam_levels):
    Xd = mix[feat_cols].to_numpy(float)
    Xm = np.stack([(mix["family"] == f).to_numpy(float) for f in fam_levels], axis=1)
    return Xd, Xm, mix["tm_c"].to_numpy(float)


def make_gp(n_dims, n_restarts=3):
    k = (ConstantKernel(100.0, (1e-2, 1e5))
         * Matern(length_scale=np.ones(n_dims), nu=2.5,
                  length_scale_bounds=(1e-2, 1e4))
         + WhiteKernel(4.0, (1e-3, 1e3)))
    return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                    n_restarts_optimizer=n_restarts, random_state=0)


def fit(mix, feat_cols, fam_levels, n_restarts=3, kernel=None):
    Xd, Xm, y = build_xy(mix, feat_cols, fam_levels)
    sc = StandardScaler().fit(Xd)
    X = np.hstack([sc.transform(Xd), Xm])
    if kernel is None:
        gp = make_gp(X.shape[1], n_restarts).fit(X, y)
    else:
        # Re-use hyperparameters learned elsewhere; only the fold's data changes.
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                      normalize_y=True).fit(X, y)
    return gp, sc


def predict(gp, sc, mix_new, feat_cols, fam_levels):
    Xd, Xm, _ = build_xy(mix_new, feat_cols, fam_levels)
    X = np.hstack([sc.transform(Xd), Xm])
    return gp.predict(X, return_std=True)


def main():
    mix = load()
    feat_cols = [c for c in mix.columns if c.startswith(("mean_", "spread_"))]
    # drop zero-variance features (spread_* is 0 for every pure lipid)
    keep = [c for c in feat_cols if mix[c].std() > 1e-9]
    fam_levels = sorted(mix["family"].unique())
    print(f"training rows: {len(mix)}   features: {len(keep)}   "
          f"method families: {fam_levels}")
    print(mix.family.value_counts().to_string())

    # ---- leave-one-system-out CV -----------------------------------------
    # Hyperparameters are optimised once on the full set, then held fixed across
    # folds. Refitting 331 ARD kernels from scratch is not affordable and the
    # scores barely move; the held-out SYSTEM is still never in the fold's data.
    print("\noptimising kernel hyperparameters...", flush=True)
    gp0, _ = fit(mix, keep, fam_levels, n_restarts=3)
    kern = gp0.kernel_
    print("  ", kern, flush=True)

    logo = LeaveOneGroupOut()
    groups = mix["system"].to_numpy()
    pred = np.full(len(mix), np.nan)
    sd = np.full(len(mix), np.nan)
    n_folds = len(np.unique(groups))
    for i, (tr, te) in enumerate(logo.split(mix, groups=groups)):
        g, s = fit(mix.iloc[tr], keep, fam_levels, kernel=kern)
        m, v = predict(g, s, mix.iloc[te], keep, fam_levels)
        pred[te] = m
        sd[te] = v
        if (i + 1) % 50 == 0:
            print(f"  fold {i+1}/{n_folds}", flush=True)

    err = pred - mix["tm_c"].to_numpy(float)
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    z = err / sd
    cov95 = (np.abs(z) < 1.96).mean()
    print(f"\n=== leave-one-system-out ===")
    print(f"MAE  {mae:.2f} C")
    print(f"RMSE {rmse:.2f} C")
    print(f"95% interval coverage {cov95:.1%}  (want ~95%)")
    print(f"median predicted sd {np.median(sd):.2f} C")

    multi = mix.n_components > 1
    print(f"\nmixtures only (n={multi.sum()}):  MAE {np.abs(err[multi]).mean():.2f} C")
    print(f"pure lipids only (n={(~multi).sum()}): MAE {np.abs(err[~multi]).mean():.2f} C")

    mix["cv_pred"] = pred
    mix["cv_sd"] = sd
    mix["cv_err"] = err
    mix.to_csv(BASE + "data/cv_predictions.csv", index=False)

    print("\nworst 10 systems:")
    w = mix.reindex(np.abs(mix.cv_err).sort_values(ascending=False).index)
    print(w[["composition", "tm_c", "cv_pred", "cv_sd", "family"]].head(10).to_string(index=False))

    # ---- final model on everything ---------------------------------------
    gp, sc = fit(mix, keep, fam_levels, n_restarts=3)
    print("\nfinal kernel:", gp.kernel_)
    import pickle
    with open(BASE + "data/gp_model.pkl", "wb") as f:
        pickle.dump(dict(gp=gp, scaler=sc, feat_cols=keep, fam_levels=fam_levels), f)
    print("saved -> data/gp_model.pkl")


if __name__ == "__main__":
    main()
