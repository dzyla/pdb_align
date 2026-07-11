# SP1 Accessor Mapping (GUI-computed → API accessor)

This is SP2's build checklist: every quantity the current Streamlit app derives
from `pdb_align.core` must be read from these API accessors instead. When SP2 is
done, `struct_pair_align.py` imports nothing from `pdb_align.core`, and
`tests/test_convergence_guard.py` flips from xfail to a strict pass.

| GUI need                         | API accessor                                   |
|----------------------------------|------------------------------------------------|
| Run alignment (any mode)         | `PDBAligner.align(mode, strategy, atoms, ...)` |
| Per-residue RMSD table           | `AlignmentResult.get_rmsd_df(on=...)`          |
| Top deviation peaks              | `AlignmentResult.report_peaks(...)`            |
| Aligned coords (ref/mob)         | `AlignmentResult.get_aligned_coords()`         |
| Transformed structure for 3D     | `AlignmentResult.aligned_structure(color_by)`  |
| Matched residue pairs            | `AlignmentResult.get_matched_pairs()`          |
| Summary numbers (rmsd/tm/cov)    | `AlignmentResult.summary_stats()`              |
| Per-chain RMSD                   | `AlignmentResult.per_chain`                    |
| Flexible domains / hinges        | `AlignmentResult.domains`                      |
| Sequence alignment text          | `AlignmentResult.get_sequence_alignment()`     |
| Quality verdict / flags          | `AlignmentResult.quality`                      |
| Downloads (zip, scripts, report) | `AlignmentResult.export_bundle(...)`           |
| Ensemble tables/plots            | `EnsembleResult.*` + `.export_bundle(...)`     |

No remaining GUI quantity requires importing `pdb_align.core`.

## Notes for SP2

- `aligned_structure(color_by=...)` returns an in-memory `gemmi.Structure`; the
  GUI can write it to a temp PDB for Py3Dmol or read coords directly.
- `quality` gives the plain-language verdict, band, confidence, `flagged_regions`
  (contiguous high-RMSD/hinge runs), and `warnings` — surface these prominently.
- `export_bundle(path, include=[...], fmt="zip"|"dir")` replaces the app's
  `export_zip_*` / `_generate_pymol_script` helpers.
