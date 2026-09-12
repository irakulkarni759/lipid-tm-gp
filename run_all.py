#!/usr/bin/env python3
"""Rebuild everything from the source spreadsheet. One command, correct order.

    python3 run_all.py

Takes a few minutes; the cross-validation is the slow part.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("build_dataset.py",      "parse the spreadsheet into a tidy table"),
    ("featurize.py",          "SMILES -> structural descriptors (first pass)"),
    ("apply_qc.py",           "flag mislabelled and inconsistent records"),
    ("featurize.py",          "rebuild descriptors on the QC'd data"),
    ("model.py",              "fit the GP and cross-validate"),
    ("add_offset.py",         "compute per-instrument offsets"),
    ("make_final_tables.py",  "write lipids.csv and measurements.csv"),
    ("plots.py",              "render the figures"),
]

for i, (script, what) in enumerate(STEPS, 1):
    print(f"\n[{i}/{len(STEPS)}] {script}  ({what})", flush=True)
    r = subprocess.run([sys.executable, "-W", "ignore",
                        os.path.join(ROOT, "src", script)])
    if r.returncode != 0:
        sys.exit(f"failed at {script}")

print("\ndone. see data/measurements.csv, data/lipids.csv, figures/")
