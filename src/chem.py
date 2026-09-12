"""Per-lipid structural descriptors extracted from SMILES.

Tm is driven overwhelmingly by two things: what the acyl chains look like, and
what the headgroup is. Generic whole-molecule RDKit descriptors (MolWt, LogP,
TPSA) only smear those together, so this module pulls them out explicitly.

Chains are found by severing each ester/amide bond and taking the fragment that
hangs off the carbonyl. Headgroup is assigned by SMARTS.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]")
AMIDE = Chem.MolFromSmarts("[CX3](=O)[NX3][#6]")

# Written tightly on purpose. A loose pattern like "COP(=O)(O)OCC(O)CO" for PG
# also matches the glycerol BACKBONE of any diacyl phospholipid, because SMARTS
# "O" happily matches an ester oxygen. Free hydroxyls are therefore pinned with
# explicit [OX2H1], and the phosphate is matched charge-agnostically since the
# source sheet writes it both ionised and neutral.
_P = "[#15](=O)([OX2H1,OX1-,OX2H0])[OX2]"
HEADGROUPS = [
    ("CL",     "[#15](=O)[OX2][CH2][CH1]([OX2H1])[CH2][OX2][#15](=O)"),
    ("SM",     _P + "[CH2][CH2][N+]([CH3])([CH3])[CH3]"),   # refined below by ester test
    ("PC",     _P + "[CH2][CH2][N+]([CH3])([CH3])[CH3]"),
    ("PS",     _P + "[CH2][CH1]([NX3])[CX3](=O)[OX2H1,OX1-]"),
    ("PG",     _P + "[CH2][CH1]([OX2H1])[CH2][OX2H1]"),
    ("PE",     _P + "[CH2][CH2][NX4,NX3]"),
    ("PI",     _P + "[CH1]1[CH1]([OX2H1])[CH1]([OX2H1])[CH1]([OX2H1])[CH1]([OX2H1])[CH1]1[OX2H1]"),
    ("PA",     "[CH2][OX2][#15](=O)([OX2H1,OX1-])[OX2H1,OX1-]"),
    ("glyco",  "[CX4][OX2][CX4H1]1[OX2][CX4H1][CX4H1]([OX2H1])[CX4H1]([OX2H1])[CX4H1]1[OX2H1]"),
    ("sterol", "C1CC2CCC3C(CCC4CCCCC34)C2C1"),
]
_HG = [(n, Chem.MolFromSmarts(sm)) for n, sm in HEADGROUPS]

HG_CLASSES = ["PC", "PE", "PS", "PG", "PA", "PI", "CL", "SM",
              "ceramide", "glyco", "sterol", "other"]


def headgroup(mol):
    # Sphingomyelin carries a PC headgroup on an amide-linked sphingoid base,
    # so it is only distinguishable from PC by the absence of ester linkages.
    sphingoid = mol.HasSubstructMatch(AMIDE) and not mol.HasSubstructMatch(ESTER)
    for name, patt in _HG:
        if patt is None or not mol.HasSubstructMatch(patt):
            continue
        if name == "SM":
            if sphingoid:
                return "SM"
            continue
        if name == "PC" and sphingoid:
            return "SM"
        return name
    if sphingoid:
        return "ceramide"
    return "other"


def _db_positions(frag):
    """Double-bond positions counted from the carbonyl carbon (Delta numbering)."""
    carbonyl = None
    for b in frag.GetBonds():
        if (b.GetBondType() == Chem.BondType.DOUBLE and
                {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} == {"C", "O"}):
            carbonyl = (b.GetBeginAtomIdx() if b.GetBeginAtom().GetSymbol() == "C"
                        else b.GetEndAtomIdx())
            break
    if carbonyl is None:
        return []
    # BFS distance from the carbonyl carbon over carbon atoms only
    dist = {carbonyl: 1}
    stack = [carbonyl]
    while stack:
        cur = stack.pop(0)
        for nb in frag.GetAtomWithIdx(cur).GetNeighbors():
            if nb.GetSymbol() == "C" and nb.GetIdx() not in dist:
                dist[nb.GetIdx()] = dist[cur] + 1
                stack.append(nb.GetIdx())
    pos = []
    for b in frag.GetBonds():
        if (b.GetBondType() == Chem.BondType.DOUBLE and
                b.GetBeginAtom().GetSymbol() == "C" and b.GetEndAtom().GetSymbol() == "C"):
            d = [dist.get(b.GetBeginAtomIdx()), dist.get(b.GetEndAtomIdx())]
            d = [x for x in d if x is not None]
            if d:
                # cis puts a ~30 degree kink in the chain and wrecks packing;
                # trans is nearly straight and behaves much like saturated.
                # 18:1c d9 PE melts at -16 C, 18:1t d9 PE at +38 C.
                st = b.GetStereo()
                is_cis = st in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS)
                is_trans = st in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS)
                pos.append((min(d), 1 if is_cis else (-1 if is_trans else 0)))
    return sorted(pos)


def acyl_chains(mol):
    """Return [(n_carbons, n_double_bonds, [db positions]), ...] for each acyl chain.

    Double-bond POSITION matters as much as count. A cis bond mid-chain (18:1 d9,
    DOPC, Tm -17 C) destroys packing; the same bond near a chain end (18:1 d2,
    Tm +41 C) barely perturbs it. Without position the model cannot tell those
    apart and splits the difference on both.
    """
    bonds = []
    for patt in (ESTER, AMIDE):
        for match in mol.GetSubstructMatches(patt):
            c, _o, x = match[0], match[1], match[2]
            b = mol.GetBondBetweenAtoms(c, x)
            if b is not None and not b.IsInRing():
                bonds.append(b.GetIdx())
    if not bonds:
        return []
    frag_mol = Chem.FragmentOnBonds(mol, sorted(set(bonds)), addDummies=False)
    chains = []
    for frag in Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False):
        atoms = list(frag.GetAtoms())
        carbons = [a for a in atoms if a.GetSymbol() == "C"]
        if len(carbons) < 2 or any(a.GetSymbol() in ("P", "N") for a in atoms):
            continue
        if any(a.IsInRing() for a in atoms):
            continue
        # An acyl fragment keeps its carbonyl oxygen; the glycerol backbone does not.
        has_carbonyl = any(
            b.GetBondType() == Chem.BondType.DOUBLE
            and {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} == {"C", "O"}
            for b in frag.GetBonds())
        if not has_carbonyl:
            continue
        n_c = len(carbons)
        n_db = sum(1 for b in frag.GetBonds()
                   if b.GetBondType() == Chem.BondType.DOUBLE
                   and b.GetBeginAtom().GetSymbol() == "C"
                   and b.GetEndAtom().GetSymbol() == "C")
        chains.append((n_c, n_db, _db_positions(frag)))
    return chains


CHAIN_COLS = ["n_chains", "chain_mean", "chain_min", "chain_max", "chain_asym",
              "unsat_total", "unsat_mean", "frac_saturated",
              "db_pos_mean", "db_centrality", "n_cis", "n_trans", "kink"]
GLOBAL_COLS = ["mw", "logp", "tpsa", "n_rot", "n_hbd", "n_hba", "n_ring",
               "frac_csp3", "charge"]
DESC_COLS = CHAIN_COLS + GLOBAL_COLS + [f"hg_{h}" for h in HG_CLASSES]


def describe(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ch = acyl_chains(mol)
    if ch:
        lens = np.array([c for c, _, _ in ch], float)
        dbs = np.array([d for _, d, _ in ch], float)
        # centrality: 1.0 for a bond at the chain midpoint, 0.0 at either end
        cent, pmean, cis, trans = [], [], 0, 0
        for n_c, _, pos in ch:
            for pp, stereo in pos:
                pmean.append(pp)
                cent.append(1.0 - abs(pp - n_c / 2.0) / (n_c / 2.0))
                cis += stereo == 1
                trans += stereo == -1
        chain = dict(
            n_chains=len(ch), chain_mean=lens.mean(), chain_min=lens.min(),
            chain_max=lens.max(), chain_asym=lens.max() - lens.min(),
            unsat_total=dbs.sum(), unsat_mean=dbs.mean(),
            frac_saturated=float((dbs == 0).mean()),
            db_pos_mean=float(np.mean(pmean)) if pmean else 0.0,
            db_centrality=float(np.max(cent)) if cent else 0.0,
            n_cis=float(cis), n_trans=float(trans),
            # the packing-disruption term: cis bonds near the chain centre
            kink=float(sum(c for c, (_, st) in
                           zip(cent, [q for _, _, p in ch for q in p]) if st == 1)))
    else:
        chain = dict(n_chains=0, chain_mean=0.0, chain_min=0.0, chain_max=0.0,
                     chain_asym=0.0, unsat_total=0.0, unsat_mean=0.0,
                     frac_saturated=1.0, db_pos_mean=0.0, db_centrality=0.0,
                     n_cis=0.0, n_trans=0.0, kink=0.0)
    hg = headgroup(mol)
    out = dict(chain)
    out.update(dict(
        mw=Descriptors.MolWt(mol), logp=Crippen.MolLogP(mol),
        tpsa=rdMolDescriptors.CalcTPSA(mol),
        n_rot=rdMolDescriptors.CalcNumRotatableBonds(mol),
        n_hbd=rdMolDescriptors.CalcNumHBD(mol),
        n_hba=rdMolDescriptors.CalcNumHBA(mol),
        n_ring=rdMolDescriptors.CalcNumRings(mol),
        frac_csp3=rdMolDescriptors.CalcFractionCSP3(mol),
        charge=Chem.GetFormalCharge(mol)))
    out.update({f"hg_{h}": float(hg == h) for h in HG_CLASSES})
    out["headgroup"] = hg
    return out
