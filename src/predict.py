"""Predict Tm for an arbitrary lipid mixture, and draw the composition map."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pickle
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.join(ROOT, "src"))
from chem import describe, DESC_COLS

BASE = ROOT + os.sep

SMILES = {
    "DMPC": "CCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCC",
    "DPPC": "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC",
    "DSPC": "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCCCC",
    "DLPC": "CCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCC",
}

_model = None
_desc_cache = {}


def load_model():
    global _model
    if _model is None:
        with open(BASE + "data/gp_model.pkl", "rb") as f:
            _model = pickle.load(f)
    return _model


def _desc(smiles):
    if smiles not in _desc_cache:
        _desc_cache[smiles] = describe(smiles)
    return _desc_cache[smiles]


def features(components):
    """components: list of (smiles, mole_fraction). Returns the mixture feature dict."""
    x = np.array([f for _, f in components], float)
    x = x / x.sum()
    D = np.array([[_desc(s)[c] for c in DESC_COLS] for s, _ in components], float)
    mean = x @ D
    spread = x @ np.abs(D - mean)
    out = {f"mean_{c}": v for c, v in zip(DESC_COLS, mean)}
    out.update({f"spread_{c}": v for c, v in zip(DESC_COLS, spread)})
    return out


def predict_tm(components, family="calorimetry"):
    """Predicted Tm in C and 1 s.d. uncertainty.

    family: which instrument the answer should be phrased as. 'calorimetry' gives
    the DSC-equivalent number comparable to published values; 'laurdan_plate'
    gives what Aria's plate reader would be expected to report.
    """
    m = load_model()
    row = features(components)
    Xd = np.array([[row[c] for c in m["feat_cols"]]], float)
    Xm = np.array([[1.0 if f == family else 0.0 for f in m["fam_levels"]]])
    X = np.hstack([m["scaler"].transform(Xd), Xm])
    mu, sd = m["gp"].predict(X, return_std=True)
    return float(mu[0]), float(sd[0])


def binary_scan(lipid_a, lipid_b, n=101, family="calorimetry"):
    fracs = np.linspace(0, 1, n)
    mus, sds = [], []
    for f in fracs:
        comps = []
        if f < 1:
            comps.append((SMILES[lipid_b], 1 - f))
        if f > 0:
            comps.append((SMILES[lipid_a], f))
        mu, sd = predict_tm(comps, family=family)
        mus.append(mu)
        sds.append(sd)
    return fracs, np.array(mus), np.array(sds)
