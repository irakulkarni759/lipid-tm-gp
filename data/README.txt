Two files. One shape each. Nothing else to read.

lipids.csv        THE DICTIONARY. One row per lipid.
                  name, SMILES, and the structural facts read out of that
                  SMILES: headgroup, tail lengths, double bonds, cis/trans.

measurements.csv  THE DATA. One row per measurement, always.
                  A 2-lipid mixture is ONE row, with lipid_1/lipid_2 and their
                  mol percentages side by side. SMILES sit inline next to each
                  lipid so you never have to look anything up.

                  tm_measured_c       what the instrument reported
                  instrument_offset_c how much that instrument reads high
                  tm_corrected_c      measured minus offset. the comparable one.
                  trusted             False = we flagged it as unreliable

Scope: the 28 lipid mixtures, plus the 39 pure-lipid readings for the lipids
that appear in those mixtures (a 100/0 point anchors its series). The ~380
unrelated pure lipids from the literature compilation are training ballast for
the model and are not in here.

archive/ holds the intermediate files from building these. Ignore it.
