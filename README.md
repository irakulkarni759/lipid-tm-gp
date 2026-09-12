# Predicting lipid melting temperature from composition

Predicts the gel-to-fluid transition temperature (Tm) of a lipid mixture from
the chemical structures of its components and their mole fractions, and reports
how confident it is in every prediction.

Built around a DMPC/DPPC plate-reader series measured in house, combined with
~400 Tm values compiled from the literature.

## Quick start

```bash
pip install -r requirements.txt
python3 run_all.py              # rebuilds everything from the spreadsheet
```

Ask it a question from the command line:

```bash
python3 ask.py DMPC 70 DPPC 30          # what does this melt at?
python3 ask.py --target 42 DPPC DSPC    # what melts at 42 C?
python3 ask.py --list                   # which lipids does it know?
python3 ask.py --find SM                # search by name
```

Add `--plate` for the plate-reader scale instead of the DSC scale that published
values use. 341 lipids are available; mixtures may have any number of components.

Or from Python:

```python
import sys; sys.path.insert(0, "src")
from predict import predict_tm, SMILES

predict_tm([(SMILES["DMPC"], 0.65), (SMILES["DPPC"], 0.35)],
           family="calorimetry")     # -> (33.9, 6.6)  degrees C, 1 s.d.
```

## Which lipids you can ask about

`ask.py` accepts 341 lipids, but they are not equally supported. Three tiers.

**Tier 1, trust these.** Measured in real mixtures, so mixture predictions are
on solid ground.

| lipid | measurements behind it |
|---|---|
| DPPC | 38 |
| DMPC | 28 |
| DOPC | 7 |
| DSPC | 7 |
| POPC | 7 |
| Cholesterol | 5 |
| DMPG | 3 |
| PSM | 1 |
| SSM | 1 |

**Tier 2, ask but read the interval.** One or two measurements each, always as a
*pure* lipid, never mixed with anything. The model knows roughly where each melts
on its own; any mixture containing one is an extrapolation.

- PC: DLPC
- PE: DLPE, DMPE, DPPE, DSPE, DOPE, POPE
- PG: DLPG, DPPG, DSPG, DOPG, POPG
- PS: DLPS, DMPS, DPPS, DSPS, DOPS, POPS
- PA: DLPA, DMPA, DPPA, DSPA, DOPA, POPA

**Tier 3, the long tail.** ~300 more from the Marsh compilation: odd chain
lengths, cardiolipins, glycolipids, sphingomyelins. Findable, but thin.

```bash
python3 ask.py --find PG       # search by name
python3 ask.py --find 18:1     # or by chain spec
```

### Naming, if the abbreviations are unfamiliar

First letter is tail length (**L**=12, **M**=14, **P**=16, **S**=18 carbons;
**O** and **PO** mean a kinked unsaturated tail). The last two letters are the
headgroup (**PC**, **PE**, **PG**, **PS**, **PA**). So DPPC is 16-carbon tails
with a PC head.

### Try it

```bash
python3 ask.py DMPC 60 DPPC 40      # two Tier 1 lipids
python3 ask.py DPPC 50 DPPG 50      # Tier 1 + Tier 2
```

The second interval is noticeably wider. That widening is the model reporting
that it is on thinner ice, and it is the most useful thing it does.


## How well it works

Validated by leave-one-system-out, holding out an entire lipid system at a time.
A random train/test split would score far better and mean nothing, because
replicate measurements of the same system from the same paper would land on
both sides.

| metric | value |
|---|---|
| mean absolute error | **3.9 °C** |
| RMSE | 6.2 °C |
| 95% interval coverage | 95.8% (target 95%) |
| mixtures only (n=23) | 5.6 °C |
| pure lipids only (n=382) | 3.8 °C |

Recovers values it had to reason toward: pure DMPC 23.5 (published 23.9), pure
DPPC 41.3 (41.4), DMPC/DSPC 50:50 41.3 (42.0).

The coverage figure is the one that matters most. When the model says ±5 °C it
is right 95% of the time, so the error bars can be trusted for deciding what to
measure next.

## Output

| file | contents |
|---|---|
| `data/lipids.csv` | one row per lipid: SMILES, headgroup, tail lengths, cis/trans |
| `data/measurements.csv` | one row per measurement: composition, Tm, instrument offset, corrected Tm |
| `figures/dmpc_dppc_binary.png` | prediction across the composition axis with uncertainty band |
| `figures/validation.png` | predicted vs measured, and the calibration check |

## How it works

1. **Parse** the source spreadsheet, where composition is stored as text inside
   cells (`'DPPC (16:0) 80%'`), into one row per measurement.
2. **Featurize by structure, not by name.** Each lipid's SMILES becomes numbers
   (acyl chain lengths, double-bond count, position and cis/trans geometry,
   headgroup class). Components are combined by mole-fraction weighting, plus
   "spread" terms that capture how mismatched the components are. Without the
   spread terms the model could only ever draw a straight line between the two
   pure-lipid values. A name-based encoding could not extrapolate to unseen
   lipids at all.
3. **Model the instrument.** Method enters the kernel as one-hot columns so the
   GP learns a per-instrument offset rather than averaging incompatible
   thermometers. The learned plate-reader offset is **+8.9 °C** versus
   calorimetry, matching the raw gap between our pure DMPC (32.48 °C) and the
   literature DSC value (23.9 °C).
4. **Fit** a Gaussian process with an ARD Matérn kernel.

## Data quality

The source compilation has problems that quietly wreck a naive fit. `src/qc.py`
catches them automatically and quarantines 53 records, each with a stated reason
in `identity_reason`.

1. **One structure claimed by several molecules.** SMILES were assigned by lipid
   *label* while the Notes column carries the real identity in Marsh notation.
   Three rows labelled DPPC carry the DPPC structure but are actually
   (16:0)&#8322;PC at 41.4 °C, (16:0)&#8322;PC-chol at 24.0 °C, and
   (16:0/19:cPrc&#916;9)PC at &minus;10.0 °C. 18 structures were affected.
2. **Counterion and ionisation state not encoded.** (16:0)&#8324;CL spans
   39.5&ndash;88.3 °C across Na&#8322;/K&#8322;/NH&#8324;/Ca/Mg forms with
   byte-identical features.
3. **Replicate disagreement.** DPPS appears at 54.0, 53.9 and 155.0 °C.
4. **4 lipids have no structure anywhere** (DSPE-PEG2kDa, HSPC, Lyso PC, MSPC),
   excluding 23 measurements including the PEGylated liposome series.
   DSPE-PEG2kDa is a polymer and is not cleanly representable as one SMILES.

Fixing 2 and 3 upstream would recover ~30 measurements.

## Limits

- **Only 28 mixtures exist in the whole dataset**, 15 of them DMPC/DPPC, and
  mixtures are the target application. This is the real bottleneck; more binary
  series is the highest-value next experiment.
- Cholesterol has no acyl chains, so cholesterol-containing mixtures get very
  wide intervals (s.d. ~27 °C). The model correctly reports that it does not know.
- A binary mixture has a solidus-liquidus band, not a single Tm. What is
  modelled is the first-derivative peak of the Laurdan GP curve.
- The instrument offset is fitted as one constant per method. It is well
  determined at the pure endpoints and noisier across the middle of the
  composition axis, so treat `tm_corrected_c` as a working number.

## Layout

```
run_all.py              rebuild everything, correct order
src/build_dataset.py    spreadsheet -> tidy table, SMILES backfill, mole fractions
src/chem.py             SMILES -> chains, headgroup, stereochemistry
src/qc.py               identity and consistency checks
src/apply_qc.py         runs the QC pass
src/featurize.py        per-lipid descriptors -> mixture features
src/model.py            GP fit + leave-one-system-out CV
src/add_offset.py       per-instrument offsets
src/make_final_tables.py  writes lipids.csv and measurements.csv
src/predict.py          predict_tm() and binary_scan()
src/plots.py            figures
data/raw/               the source spreadsheet (tracked)
```
