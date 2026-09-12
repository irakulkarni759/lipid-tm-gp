import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(ROOT, "src"))
from predict import binary_scan, SMILES, predict_tm

BASE = ROOT + os.sep
plt.rcParams.update({"figure.dpi": 140, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

ARIA = pd.DataFrame({
    "x_dmpc": [1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0],
    "tm": [32.48, 35.28, 37.14, 38.39, 39.81, 39.95, 38.99, 41.21, 37.28, 38.03, 37.58],
    "reliable": [True]*5 + [False]*6,
})


def fig_binary():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))

    for ax, fam, title in [
            (axes[0], "calorimetry", "DSC-equivalent scale\n(comparable to published values)"),
            (axes[1], "laurdan_plate", "Plate-reader Laurdan GP scale\n(what Aria's assay reports)")]:
        f, mu, sd = binary_scan("DMPC", "DPPC", family=fam)
        x = 1 - f          # x axis = mole fraction DPPC
        order = np.argsort(x)
        x, mu, sd = x[order], mu[order], sd[order]
        ax.fill_between(x, mu - 1.96*sd, mu + 1.96*sd, alpha=.18,
                        color="#3b7dd8", lw=0, label="95% credible interval")
        ax.plot(x, mu, color="#1f4e9c", lw=2, label="GP prediction")

        lit = pd.DataFrame({"x": [0, .2, .5, .5, .5, .6, .8, 1.0],
                            "tm": [23.9, 26.4, 31.0, 31.5, 34.0, 34.7, 37.3, 41.4]})
        ax.scatter(lit.x, lit.tm, marker="s", s=34, facecolor="white",
                   edgecolor="#333", zorder=5, label="literature (DSC / probe)")
        r = ARIA[ARIA.reliable]
        u = ARIA[~ARIA.reliable]
        ax.scatter(1-r.x_dmpc, r.tm, marker="o", s=46, color="#d1345b",
                   zorder=6, label="Aria, reliable (100/0-60/40)")
        ax.scatter(1-u.x_dmpc, u.tm, marker="x", s=40, color="#bbb",
                   zorder=6, label="Aria, flagged unreliable")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("mole fraction DPPC")
        ax.set_ylabel("T$_m$ (°C)")
        ax.set_xlim(-.02, 1.02)
        ax.grid(alpha=.25, lw=.5)
    axes[0].legend(fontsize=7, loc="upper left", framealpha=.95)
    fig.suptitle("DMPC / DPPC phase transition temperature, GP prediction with uncertainty",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    fig.savefig(BASE + "figures/dmpc_dppc_binary.png", bbox_inches="tight")
    print("wrote figures/dmpc_dppc_binary.png")


def fig_parity():
    cv = pd.read_csv(BASE + "data/cv_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    ax = axes[0]
    pc = cv[cv.mean_hg_PC > 0.99]
    ax.errorbar(cv.tm_c, cv.cv_pred, yerr=1.96*cv.cv_sd, fmt="none",
                ecolor="#ccc", lw=.6, zorder=1)
    ax.scatter(cv.tm_c, cv.cv_pred, s=13, color="#888", label="all lipids", zorder=2)
    ax.scatter(pc.tm_c, pc.cv_pred, s=15, color="#1f4e9c", label="PC headgroup", zorder=3)
    lo, hi = cv.tm_c.min()-8, cv.tm_c.max()+8
    ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
    ax.set_xlabel("measured T$_m$ (°C)"); ax.set_ylabel("predicted T$_m$ (°C)")
    ax.set_title("Leave-one-system-out predictions", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=.25, lw=.5)

    ax = axes[1]
    z = (cv.cv_pred - cv.tm_c) / cv.cv_sd
    ax.hist(z, bins=34, color="#3b7dd8", alpha=.8)
    for v in (-1.96, 1.96):
        ax.axvline(v, color="k", ls="--", lw=.8)
    ax.set_xlabel("z = (predicted − measured) / predicted s.d.")
    ax.set_ylabel("count")
    ax.set_title(f"Calibration: {(z.abs()<1.96).mean():.0%} inside 95% interval",
                 fontsize=9)
    ax.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    fig.savefig(BASE + "figures/validation.png", bbox_inches="tight")
    print("wrote figures/validation.png")


if __name__ == "__main__":
    fig_binary()
    fig_parity()
