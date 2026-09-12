#!/usr/bin/env python3
"""Ask the model a question. No Python needed.

    python3 ask.py DMPC 70 DPPC 30        what does this mixture melt at?
    python3 ask.py --target 40 DMPC DPPC  what mixture melts at 40 C?
    python3 ask.py --list                 which lipids does it know?
    python3 ask.py --find DPP             search for a lipid by name

Add --plate to get the answer on our plate-reader scale instead of the
calorimetry scale that published values use.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def vocabulary():
    """Every lipid the model has a structure for."""
    d = pd.read_csv(os.path.join(ROOT, "data", "lipid_descriptors.csv"))
    return dict(zip(d.lipid, d.smiles))


def resolve(name, vocab):
    if name in vocab:
        return name
    hits = [k for k in vocab if k.lower() == name.lower()]
    if len(hits) == 1:
        return hits[0]
    hits = [k for k in vocab if name.lower() in k.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sys.exit(f"Don't know a lipid called '{name}'.\n"
                 f"Try:  python3 ask.py --find {name[:3]}")
    sys.exit(f"'{name}' is ambiguous. Did you mean one of:\n  " +
             "\n  ".join(hits[:12]))


def confidence(sd):
    if sd < 5.5:
        return "confident"
    if sd < 9:
        return "reasonably sure"
    if sd < 15:
        return "not very sure"
    return "basically guessing, go measure it"


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return

    vocab = vocabulary()

    if args[0] == "--list":
        names = sorted(vocab)
        print(f"{len(names)} lipids known. Common ones:\n")
        common = ["DLPC", "DMPC", "DPPC", "DSPC", "DOPC", "POPC", "DMPG",
                  "DPPG", "DPPE", "DSPE", "Cholesterol", "PSM", "SSM"]
        for c in common:
            if c in vocab:
                print("  " + c)
        print("\nFor the full list:  python3 ask.py --find <part of a name>")
        return

    if args[0] == "--find":
        q = args[1].lower()
        hits = sorted(k for k in vocab if q in k.lower())
        print(f"{len(hits)} match '{args[1]}':")
        for h in hits[:40]:
            print("  " + h)
        if len(hits) > 40:
            print(f"  ... and {len(hits)-40} more")
        return

    from predict import predict_tm
    family = "laurdan_plate" if "--plate" in args else "calorimetry"
    args = [a for a in args if a != "--plate"]
    scale = "plate-reader scale" if family == "laurdan_plate" else "DSC scale"

    # ---- reverse mode: find the composition that hits a target ------------
    if args[0] == "--target":
        target = float(args[1])
        a, b = resolve(args[2], vocab), resolve(args[3], vocab)
        fr = np.linspace(0, 1, 201)
        tms, sds = [], []
        for f in fr:
            comps = []
            if f < 1:
                comps.append((vocab[a], 1 - f))
            if f > 0:
                comps.append((vocab[b], f))
            t, s = predict_tm(comps, family=family)
            tms.append(t); sds.append(s)
        tms = np.array(tms)
        if not (tms.min() - 1 <= target <= tms.max() + 1):
            print(f"\n{a} and {b} cannot reach {target:.0f} C in any ratio.")
            print(f"This pair only spans {tms.min():.0f} to {tms.max():.0f} C "
                  f"({scale}).")
            print("You need different lipids for that target.\n")
            return
        i = int(np.argmin(np.abs(tms - target)))
        print(f"\nTo melt at {target:.0f} C ({scale}), mix:\n")
        print(f"    {a}  {100*(1-fr[i]):.0f}%")
        print(f"    {b}  {100*fr[i]:.0f}%")
        print(f"\nPredicted {tms[i]:.1f} C, give or take {sds[i]:.1f}.")
        print(f"The model is {confidence(sds[i])}.\n")
        return

    # ---- forward mode: what does this mixture melt at? --------------------
    if len(args) % 2 != 0:
        sys.exit("Give lipids and percentages in pairs, e.g.\n"
                 "  python3 ask.py DMPC 70 DPPC 30")
    names = [resolve(args[i], vocab) for i in range(0, len(args), 2)]
    pcts = [float(args[i]) for i in range(1, len(args), 2)]
    total = sum(pcts)
    if abs(total - 100) > 1:
        print(f"(percentages add to {total:.0f}, rescaling to 100)")
    comps = [(vocab[n], p / total) for n, p in zip(names, pcts)]

    tm, sd = predict_tm(comps, family=family)
    print("\n  " + "  /  ".join(f"{n} {100*p/total:.0f}%"
                                for n, p in zip(names, pcts)))
    print(f"\n    melts at about {tm:.1f} C, give or take {sd:.1f}")
    print(f"    likely between {tm-1.96*sd:.0f} and {tm+1.96*sd:.0f} C")
    print(f"    the model is {confidence(sd)}")
    print(f"    ({scale})\n")


if __name__ == "__main__":
    main()
