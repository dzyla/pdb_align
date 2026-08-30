# SP3 — Scientific correctness + interface metrics (DockQ / epitope / model evaluation)

Goal: make every reported number defensible against the published reference
definition, and add the interface-metrics layer (DockQ, epitope diagnostics,
prediction-vs-template model evaluation) that turns pdb_align into a
prediction-assessment tool.

## Verified external constants (sources fetched 2026-08-29)

- **TM p-value** — Xu & Zhang, Bioinformatics 2010 (26:889): TM-scores of
  random gapless pairs follow EVD F(x)=exp(-exp(-(x-mu)/sigma)) with
  mu=0.1512, sigma=0.0242, length-independent. P(TM>=x)=1-F(x).
  Golden value: P(TM>=0.5) = 5.5e-7 (stated in paper).
- **DockQ** — Basu & Wallner 2016; definitions cross-checked against
  github.com/bjornwallner/DockQ source: fnat contacts = any heavy-atom pair
  < 5 A across interface; interface residues for iRMSD = any heavy-atom pair
  < 10 A in the NATIVE; iRMSD/LRMSD over backbone (N, CA, C, O); LRMSD after
  superposing receptor backbone; DockQ = (fnat + 1/(1+(iRMSD/1.5)^2)
  + 1/(1+(LRMSD/8.5)^2))/3; classes: <0.23 incorrect, <0.49 acceptable,
  <0.80 medium, >=0.80 high.
- **pDockQ** — Bryant, Pozzati & Elofsson 2022 (FoldDock src/pdockq.py):
  contacts = CB (CA for Gly) pairs <= 8 A between the two groups;
  x = mean(interface pLDDT over unique interface residues of both sides)
  * log10(n_contacts); pDockQ = 0.724/(1+exp(-0.052*(x-152.611))) + 0.018.

## Tasks

1. metrics.py: replace ad-hoc Gumbel p-value with Xu & Zhang EVD; add
   `tm_optimal_superposition` (TM-score-maximizing iterative superposition,
   TMscore-program style: fragment seeds + iterate subset d<d_cut Kabsch,
   keep max TM); chunked lDDT-Ca.
2. GDT_TS: compute over ALL matched pairs (not inliers) and normalize by the
   reference selection's residue count (`n_total`), CASP-style; label
   "single superposition". `compute_gdt_ts(dists, n_total=None)`.
3. Rename `compute_cad_score_approx` -> `compute_contact_overlap` (it is a
   Ca contact-map Jaccard, not CAD); deprecated alias retained.
4. Wire lDDT-Ca into `summary_stats()["lddt_ca"]` + report.
5. `AlignmentResult.get_tm_score` uses TM-optimal superposition (reported TM
   is the max over superpositions for the given residue correspondence, as
   TM-align does); RMSD superposition still drives coordinates/plots.
6. NEW `pdb_align/interface.py`: `compute_dockq`, `epitope_metrics`
   (precision/recall/F1/Jaccard of epitope+paratope residue sets),
   `evaluate_antibody_complex` (H+L grouped receptor), `compute_pdockq`.
   Residue correspondence via per-chain sequence alignment (robust to
   numbering mismatches); chain groups mapped by Hungarian identity with
   permutation search over near-identical chains maximizing fnat.
7. NEW `pdb_align/evaluate.py`: `evaluate_models(ref, models, ...)` ranked
   DataFrame (rmsd/tm/gdt/lddt/coverage [+ dockq/fnat/irmsd/lrmsd/capri/
   epitope-F1/pdockq]) + report; CLI `--models`, `--receptor-chains`,
   `--ligand-chains`, `--antibody-chains`, `--antigen-chains`.
8. Hygiene: pyproject/setup.py consolidation (gemmi + matplotlib + scripts
   entry, delete setup.py); CLI `--version` + except fix; `_coverage_score`
   dedup; curated modified-residue map (SEP/TPO/PTR/...); seq-free
   all_heavy restricted to backbone across unlike residue types.
9. Tests: analytic golden tests for every metric (DockQ=1 identity; pure
   ligand translation -> exact LRMSD; fnat by construction; pDockQ sigmoid;
   p-value 5.5e-7 at TM=0.5; GDT normalization) + cross-check vs pip DockQ
   when installed (skipif).
