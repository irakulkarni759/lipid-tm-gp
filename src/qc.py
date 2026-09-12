"""Identity QC.

The Marsh-handbook rows in the source workbook were given SMILES by lipid *label*,
but the Notes column carries the real molecule in Marsh notation. In 18 cases one
SMILES is shared by chemically different species - different counterion state,
alpha vs beta anomer, ether vs ester linkage, cholesterol/biotin conjugates. Those
differ in Tm by up to 50 C while carrying byte-identical features, which a GP can
only absorb as noise.

This module checks each row's stated identity against the structure it was given
and flags the ones that disagree.
"""
import re
import numpy as np
import pandas as pd

# Tokens that mark a molecule the plain chain notation does not describe:
# salt/ionisation codes, conjugates, ether linkages, lyso forms.
# Deliberately narrow. Double-bond position markers (18:1c/\u22069) are legitimate
# distinct molecules with their own SMILES and must NOT be flagged here; a
# genuinely mismatched one is caught by the chain-consistency test instead.
#
# CONJUGATE = a chemically different molecule wearing the same label. The
# structure we hold is wrong for it, so it stays quarantined.
CONJUGATE = re.compile(r"(-chol|biotin|Lys|\bO-|\bdiether|\blyso)", re.IGNORECASE)

# COUNTERION = the SAME lipid with a different salt form. The structure is
# right, only the counterion differs, and Marsh records it after a dot. These
# are recoverable once the counterion is passed to the model as a feature.
COUNTERION = re.compile(r"\.\(?(H|Li|Na|K|NH4|Rb|Cs|Mg|Ca|Ba|Sr|Tris)\)?(\d*)[a-z]?$",
                        re.IGNORECASE)
MODIFIER = CONJUGATE
# Trailing bare 0 / 9.0 style codes are Marsh's pH / ionisation-state markers.
STATE_CODE = re.compile(r"\d+\.\d+$|(?<=[A-Za-z])0$")
CHAIN = re.compile(r"(\d+):(\d+)")


def notation_chains(notation):
    """Total acyl carbons and double bonds implied by a Marsh notation.

    '(16:0)2PC'      -> (32, 0)   two identical 16:0 chains
    '(16:0/18:1)PC'  -> (34, 1)
    '(16:0)4CL'      -> (64, 0)
    """
    if not isinstance(notation, str):
        return None
    head = notation.split(")")[0] + ")" if ")" in notation else notation
    chains = CHAIN.findall(head)
    if not chains:
        return None
    mult = re.search(r"\)(\d+)", notation)
    n = int(mult.group(1)) if mult and len(chains) == 1 else 1
    carbons = sum(int(c) for c, _ in chains) * n
    doubles = sum(int(d) for _, d in chains) * n
    return carbons, doubles


def run(long, lipdesc):
    """Annotate `long` with identity QC columns. Returns the annotated frame."""
    df = long.copy()
    df["marsh"] = df["notes"].astype(str).str.extract(
        r"Marsh notation:\s*(.+?)\s*$")[0]

    d = lipdesc.set_index("lipid")
    # Exact acyl-carbon total now available from the chain parser.
    df["chainc_ref"] = df["lipid"].map(d["chain_mean"] * d["n_chains"])
    df["nunsat_ref"] = df["lipid"].map(d["unsat_total"])

    flags, reasons = [], []
    for row in df.itertuples(index=False):
        notation = row.marsh
        if not isinstance(notation, str) or not notation.strip():
            flags.append(False); reasons.append(""); continue
        if CONJUGATE.search(notation):
            flags.append(True); reasons.append("conjugate, structure differs"); continue
        if COUNTERION.search(notation):
            # same molecule, different salt. recoverable; counterion becomes a feature.
            flags.append(False); reasons.append(""); continue
        if STATE_CODE.search(notation):
            # Marsh ionisation-state codes. Every such row in this sheet carries
            # Tm exactly 0.0, which is a missing-value placeholder, not a
            # measurement, so these stay out.
            flags.append(True); reasons.append("ionisation-state placeholder"); continue
        nc = notation_chains(notation)
        if nc is None or np.isnan(row.chainc_ref):
            flags.append(False); reasons.append(""); continue
        carbons, doubles = nc
        # Both sides are now true acyl-carbon totals, so they should agree
        # exactly; 1 carbon of slack covers odd backbone conventions.
        if abs(row.chainc_ref - carbons) > 1 or row.nunsat_ref != doubles:
            flags.append(True)
            reasons.append(f"notation {notation} != structure "
                           f"(C{carbons}:{doubles} vs C{row.chainc_ref:.0f}:{row.nunsat_ref:.0f})")
        else:
            flags.append(False); reasons.append("")

    df["identity_suspect"] = flags
    df["identity_reason"] = reasons

    # Residual check: after the above, any SMILES still claimed by two different
    # Marsh notations is an unresolved collision (e.g. alpha vs beta anomers,
    # which share a formula and are indistinguishable to these descriptors).
    ok = df[~df.identity_suspect].dropna(subset=["marsh", "smiles"]).copy()
    # strip the salt suffix first, so Na2/K2/Ca forms of one lipid are not
    # mistaken for different molecules sharing a structure
    ok["base"] = ok["marsh"].str.replace(COUNTERION, "", regex=True)
    n_names = ok.groupby("smiles")["base"].nunique()
    collided = set(n_names[n_names > 1].index)
    hit = df["smiles"].isin(collided) & ~df["identity_suspect"]
    df.loc[hit, "identity_suspect"] = True
    df.loc[hit, "identity_reason"] = "one structure claimed by several distinct lipids"

    # Replicate consistency. The same pure lipid is often reported several times
    # under conditions the sheet does not record (pH, counterion, hydration).
    # DPPS appears at 54.0, 53.9 and 155.0 C; DMPS at 35.0 and 91.6. Those are not
    # measurement noise, they are different physical states sharing one feature
    # vector, and a GP can only absorb them by inflating its noise term until every
    # prediction is useless. Flag values far from the median of their own replicates.
    pure = df[(df["n_components"] == 1) & df["tm_c"].notna() & ~df["identity_suspect"]]
    grp = pure.groupby("lipid")["tm_c"]
    med = grp.transform("median")
    mad = grp.transform(lambda v: (v - v.median()).abs().median())
    n = grp.transform("size")
    # 10 C floor so tight clusters do not flag ordinary scatter
    tol = np.maximum(3.0 * mad, 10.0)
    off = (n >= 3) & ((pure["tm_c"] - med).abs() > tol)
    idx = pure.index[off]
    df.loc[idx, "identity_suspect"] = True
    df.loc[idx, "identity_reason"] = "outlier vs replicates of the same lipid"

    # A measurement is suspect if any of its components is.
    df["meas_identity_ok"] = ~df.groupby("meas_id")["identity_suspect"].transform("any")
    return df.drop(columns=["chainc_ref", "nunsat_ref"])
