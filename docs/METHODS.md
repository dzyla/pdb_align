# Methods

What every number `pdb_align` reports means, how it is computed, what it is
normalized by, and where the implementation deviates from the reference. If
you are writing a methods section, this page is the source; if you are
reviewing a number, this page should tell you whether to trust it.

Cross-validation against the reference programs lives in
`tests/test_golden_crossvalidation.py` and runs in CI.

---

## 1. What is being compared: the selection

Every comparison starts from a **selection**: the set of residues that take
part. `core.select_residues()` is the only thing that builds one, and it
returns the residues *and* the one-letter sequence describing them, with the
invariant `len(sequence) == len(residues)` asserted at construction.

This matters because sequence alignment maps onto coordinates **by index**.
Any divergence between "the residues whose sequence was aligned" and "the
residues whose coordinates were used" silently mis-pairs residues and produces
a plausible, wrong RMSD. (Before v0.4.0 the two were built independently: two
identical copies of 1UBQ with two residues filtered out of one side reported
5.62 Å RMSD between a structure and itself.)

Selection rules:

- **Residues**: anything gemmi's chemical-component tables call an amino acid
  and that carries a Cα. Modified residues resolve to their parent one-letter
  code (`MSE`→M, `SEP`→S, `PTR`→Y, …) from the CCD, so they stay in the
  sequence instead of opening a phantom gap. `UNK` becomes `X`. `SEC` and
  `PYL` map to C and K because BLOSUM62 has no U or O.
- **No Cα, no residue.** A residue without a Cα cannot anchor a superposition
  or hold a sequence position, so it is excluded from both.
- **Selectors**: chain names (`"A"`), 1-based chain indices (`1`), or residue
  ranges (`"A:10-150"`). A named chain that is absent, or a range that selects
  nothing, raises — it never silently compares something else.
- **Alternate conformations**: the first altloc of each atom name wins.

### B-factor and pLDDT filters

`min_b_factor` is a lower bound on the Cα B-factor applied **symmetrically**
to both structures.

`min_plddt` is the same bound, but applied **only to a structure whose
B-factor column looks like pLDDT** (max ≤ 100, mean ≥ 50, no exact zeros).
pLDDT is a confidence where high is good; a crystallographic B-factor is a
disorder measure where *low* is good and values above 100 are routine. The two
need opposite cutoffs, so applying a pLDDT floor to an experimental reference
discards precisely the ordered core (at `--min-plddt 70`, all of it). When the
filter is skipped for a side, a warning names that side.

Filters change the denominator of everything downstream: coverage and GDT_TS
are normalized by the **filtered** reference selection, because a residue
deliberately excluded is no longer part of the target.

---

## 2. Residue correspondence

### Sequence-guided (`mode="seq_guided"`)

Semi-global Needleman–Wunsch (Biopython `PairwiseAligner`) on BLOSUM62,
gap open −10, gap extend −0.5, **end gaps free**. Free end gaps mean a domain
or a truncated construct aligns inside a longer chain without paying for the
overhang. Aligned columns with a residue on both sides become residue pairs by
position.

Known limitation, inherent to the method: a deletion inside a run of identical
residues cannot be localised. Ubiquitin has Q40–Q41; deleting either leaves
the same sequence, so the gap may be placed at 40 or 41 with identical score
and one residue ends up offset by one. A structure-based correspondence has no
such ambiguity. This is tested and documented, not worked around.

### Sequence-free (`mode="seq_free_shape"` / `"seq_free_window"`)

For pairs with too little sequence similarity to align, or where the sequence
should not be trusted:

- **shape**: each residue is described by a normalized histogram (24 bins, to
  the 98th-percentile distance) of its Cα–Cα distances to every other residue
  in the selection — a rotation- and translation-invariant descriptor.
  Residues are matched by χ² distance between descriptors through a banded
  dynamic program (band = 20% of the longer selection), which keeps the
  correspondence monotone in sequence order.
- **window**: finds the single diagonal offset of the shorter selection inside
  the longer one that minimises the L1 difference between their Cα–Cα distance
  matrices. One contiguous block, no gaps.

Both are heuristics of this package, not published methods; they are named as
such in the report (`Sequence-free (shape)`). Neither does the full
correspondence optimisation of TM-align or DALI. For a published
sequence-independent alignment, use TM-align.

Cost is O(N²) in memory (two distance matrices), so a selection above 20 000
residues is refused with an explanation rather than being killed by the OOM
killer.

### `mode="auto"`

Runs the applicable strategies and keeps the one with the higher
**coverage-weighted score**

```
score = n_pairs / (1 + (RMSD / 3 Å)²)
```

so a strategy that matches a handful of residues at near-zero RMSD cannot beat
one that superimposes the whole protein well; at equal coverage it reduces to
preferring the lower RMSD. The chosen method and the reason are on the result
(`.method`, `.reason`) and in the report. The same formula picks between the
global and local multi-chain superpositions.

---

## 3. Multi-chain correspondence

When both selections span several chains, `chains.match_chains()` finds a 1:1
chain correspondence by Hungarian assignment (`scipy.optimize.
linear_sum_assignment`) on the pairwise percent-identity matrix.

**Identity is normalized by the shorter chain.** Normalizing by alignment
length (including terminal gaps) turns a perfect 50-residue match inside a
200-residue chain into "25% identity" — a length ratio wearing an identity's
name — which breaks matching for truncated constructs, Fv fragments and
single-domain models.

**Homomultimers.** Copies of the same chain have exactly tied identity, so the
assignment among them is arbitrary and can pair reference chain A with the
wrong copy — a correspondence error that inflates RMSD with no other symptom.
When candidates are within 5% identity of each other, the mapping is refined
geometrically: superpose on the current mapping's chain centroids, reassign by
post-superposition centroid proximity, repeat until the mapping stops changing
(capped at 10 rounds, and skipped above 24 chains). The best-scoring round
wins. `mapping.refined` records whether refinement changed anything.

**Weak correspondence is flagged, not hidden.** Unrelated protein chains reach
roughly 10–20% identity by chance. A pair below 25% identity, or a pair whose
chains differ in length by more than 5×, produces a `UserWarning`, appears in
`quality.warnings` and in the text report, and caps the reported confidence at
`low` however tight the resulting fit is.

**Strategy.** Two superpositions are built and compared by the
coverage-weighted score above: *global* (Kabsch over every mapped chain's
atoms) and *local* (Kabsch over the single highest-identity chain pair).
`--strategy global|local` forces the choice. Residues are paired per chain
pair by that pair's own sequence alignment, so an unmodelled loop in one chain
shifts nothing in the others.

---

## 4. Superposition

**Kabsch** (SVD of the cross-covariance matrix, with a reflection guard so the
result is a proper rotation). Returns `R`, `t` with `R @ mobile + t` in the
reference frame, and the RMSD of that fit.

With `recycles > 0` and `keep_fraction < 1`, outliers are rejected
iteratively: refit, drop pairs beyond `max(2 Å, 1.5 × current RMSD)`, repeat
until the kept set stops changing, never dropping below `keep_fraction` of the
pairs. An RMSD obtained this way describes the kept subset, so the kept count
is reported alongside it.

**Atom selection.** `atoms="CA"` (default), `"backbone"` (N, CA, C, O) or
`"all_heavy"`. Side-chain atoms are only paired between residues **of the same
type**: an ALA CB and a TRP CB share a name but point into different
chemistry, so unlike pairs contribute backbone only.

**Per-residue values are per residue.** With several atoms per residue, the
reported per-residue deviation is the RMS over that residue's matched atoms,
and the residue count stays a residue count. (Before v0.4.0 `--atoms
all_heavy` reported 76 residues as 602 and coverage as 792%.)

---

## 5. Metrics

### RMSD

Root-mean-square deviation over matched atoms in the final superposition, in Å.

For a **flexible** (multi-domain) result, the reported RMSD combines the
per-domain RMSDs **in quadrature** over their residue counts,
`sqrt(Σ nᵢ·RMSDᵢ² / Σ nᵢ)` — RMSDs are root-mean-square quantities and an
arithmetic mean understates them (1 Å and 5 Å over equal domains is 3.6 Å, not
3.0 Å).

### TM-score — Zhang & Skolnick, *Proteins* 2004, 57:702

```
TM = (1/L) Σ 1 / (1 + (dᵢ/d₀)²),   d₀ = 1.24·(L−15)^(1/3) − 1.8,  d₀ ≥ 0.5 Å
```

Reported as TM-align reports it: the **maximum over rigid superpositions** for
the given correspondence, not the value in the RMSD-optimal frame (which
underestimates TM whenever a flexible tail drags the least-squares fit). The
search follows the reference TM-score program — seeds from every contiguous
fragment of length L, L/2, L/4 … ≥ 4 at evenly spread start positions, each
refined by re-superposing on the residues within `d₀_search` (clamped to
[4.5, 8] Å, grown by 0.5 Å when fewer than three qualify) until the selected
set stops changing.

**Deviations from TM-align.** (1) TM-align also optimises the *correspondence*
(it re-threads the alignment); we score the correspondence we were given. For
a correct correspondence the two agree to within 0.011 on our decoys; for
unrelated structures TM-align reports more, because it finds a better
alignment than the one it was handed. (2) Start positions are capped
(`max_starts=48` per fragment length) instead of exhaustive, which bounds the
cost at 0.2 s rather than 5.5 s for 3000 residues. Raising the cap cannot
lower the score — the search only keeps its best seed — and a test asserts
that monotonicity.

**Normalization.** By the **reference** selection length by default
(`normalize_by="reference"`; also `"mobile"`, `"min"`). Changing it breaks
comparability across targets.

**TM-score is a single-chain measure.** For a selection spanning several
chains the reported value is normalized by the total reference length and is
labelled `TM-score (cplx)`, with `tm_scope="complex"` in `summary_stats()`
and **per-chain TM-scores reported alongside** — one well-placed large chain
otherwise masks a badly placed small one. A complex-level value is not
comparable with a published per-chain TM-score.

### TM-score p-value — Xu & Zhang, *Bioinformatics* 2010, 26:889

P(TM_random ≥ observed) under the extreme-value distribution those authors
fitted to 7.2 × 10⁷ gapless comparisons of non-homologous domains:
`F(x) = exp(−exp(−(x−μ)/σ))`, μ = 0.1512, σ = 0.0242, length-independent by
TM-score's construction. At TM = 0.5 this gives 5.5 × 10⁻⁷, the paper's
published value. Returns 1.0 below 16 residues, where the statistic is
meaningless.

### GDT_TS — after Zemla, *NAR* 2003 (LGA)

Mean of the fractions of residues within 1, 2, 4 and 8 Å, × 100.

- Computed over **all matched pairs** in the final superposition, never an
  inlier subset: normalizing by the survivors of outlier rejection inflates
  the score.
- Normalized by the **reference selection length**, so residues that could not
  be aligned count as failures at every cutoff (CASP semantics). A model
  covering half the reference cannot exceed ~50 even if that half is perfect.
- **This is a lower bound on CASP's GDT_TS**, which maximises each cutoff's
  fraction over many superpositions via LGA. We report a single
  superposition. The report marks the value with `*` and says so.

### lDDT-Cα — Mariani *et al.*, *Bioinformatics* 2013, 29:2722

Superposition-free. Compares the two internal Cα–Cα distance matrices over all
pairs whose **reference** distance is below the 15 Å inclusion radius, scoring
the fraction preserved within 0.5, 1, 2 and 4 Å and averaging the four.
Computed in row blocks so a large complex does not allocate full N×N matrices.

Computed over the **matched** residues only; unmatched residues are not
penalised, so read it together with `coverage_pct`. This is lDDT-Cα, not
all-atom lDDT, and it does not apply the stereochemistry checks of the
reference `lddt` program.

### Contact-map overlap

Jaccard index of the two Cα contact maps (8 Å cutoff; self and i,i+1 pairs
excluded), superposition-invariant. Opt-in via
`AlignmentResult.contact_overlap()`.

**This is not CAD-score.** The CAD-score of Olechnović & Venclovas is defined
on Voronoi contact *areas* over heavy atoms. An earlier version of this
package exposed this quantity under the name `compute_cad_score_approx`; it
was never CAD-score, and the misnamed alias was removed in v0.4.0.

### DockQ — Basu & Wallner, *PLoS ONE* 2016, 11:e0161879

```
DockQ = (fnat + 1/(1+(iRMSD/1.5)²) + 1/(1+(LRMSD/8.5)²)) / 3
```

- **fnat**: fraction of native interface residue–residue contacts (any
  heavy-atom pair < 5 Å) reproduced by the model. `fnonnat` is the fraction of
  the model's contacts that are not native.
- **iRMSD**: backbone (N, CA, C, O) RMSD over the interface residues, defined
  by any heavy-atom pair within 10 Å **in the native**.
- **LRMSD**: ligand backbone RMSD after superposing on the receptor backbone.
- **CAPRI class**: incorrect < 0.23 ≤ acceptable < 0.49 ≤ medium < 0.80 ≤ high.

Receptor and ligand are given as *reference* chain names; multi-chain sides
(antibody H+L as the receptor) are treated as one merged unit, the standard
convention for immune complexes. Model chains are matched to reference chains
by sequence (Hungarian) unless given explicitly, and residues are paired per
chain pair by sequence alignment, so author numbering offsets, expression tags
and unmodelled loops do not mis-pair anything. For groups containing
sequence-identical chains, the assignment is chosen by maximising fnat over
their permutations (capped at 720 combinations).

Raises rather than guessing when the two groups share no contacts in the
reference — there is then no native interface to score.

**Cross-validated against the official `DockQ` package** on six decoys
spanning DockQ 0.02–1.00 and three CAPRI classes: fnat exact, iRMSD and LRMSD
within 1 × 10⁻⁴ Å, DockQ within 1 × 10⁻⁵.

### Epitope / paratope agreement

A site residue has any heavy atom within 4.5 Å of the other group. The model's
sites are mapped into reference numbering through the per-chain sequence
alignment, then compared as sets: precision, recall, F1, Jaccard.

This separates two failures a single DockQ number conflates: "right epitope,
imperfect pose" (low DockQ, high epitope F1) from "wrong face of the antigen"
(low DockQ, low epitope F1). `evaluate_antibody_complex()` says which in
words.

### pDockQ — Bryant, Pozzati & Elofsson, *Nat Commun* 2022, 13:1265

Reference-free interface confidence for a predicted complex. Contacts are
CB–CB pairs (CA for glycine) within 8 Å between the two groups;
`x = ⟨interface pLDDT⟩ · log₁₀(n_contacts)`;
`pDockQ = 0.724/(1+exp(−0.052·(x−152.611))) + 0.018`, constants from the
FoldDock reference implementation. Requires pLDDT in the B-factor column and
warns when the column does not look like pLDDT.

### pDockQ2 — Zhu *et al.*, *Bioinformatics* 2023, 39:btad424

Uses the PAE matrix: `X = ⟨1/(1+(PAE/10)²)⟩ · ⟨pLDDT⟩` over CB ≤ 8 Å interface
contacts, through a sigmoid with L = 1.31034849, x₀ = 84.7326239,
k = 0.0747157696, b = 0.00501886443 (reference implementation). PAE is
asymmetric, so both directions and their mean are reported. A PAE matrix whose
size does not match the residue count raises rather than being reindexed by
guesswork; non-protein tokens (ligands, nucleic acids) are not supported.

### CDR geometry — IMGT numbering (Lefranc 2003)

CDR1 27–38, CDR2 56–65, CDR3 105–117 in IMGT numbering, assigned by ANARCI.
`cdr_rmsd()` superposes the combined framework backbone with **one** Kabsch
fit and reports each CDR's backbone RMSD in that frame — a CDR fitted on
itself would hide the loop's displacement, which is usually the quantity of
interest. Kappa chains are typed `K` but labelled L1–L3.

Needs `anarci` plus HMMER 3.3.x on PATH; HMMER ≥ 3.4 changed its text output
and ANARCI's parser does not read it, which produces a clear `RuntimeError`
rather than wrong numbering.

---

## 6. Flexible (multi-domain) alignment

`mode="flexible"` runs `mode="auto"`, then looks for hinges in the per-residue
Cα deviation: a 15-residue sliding mean above 3 Å marks a hinge, whose
midpoint becomes a split; splits leaving a segment shorter than 30 residues
are dropped. Each resulting domain is refitted independently with its own
Kabsch superposition.

Two rules make the output mean something:

1. **Hinge detection runs per chain.** A chain boundary is not a hinge, and a
   "domain" spanning one is not a rigid body — fitting it mixes two
   independent motions. (Before v0.4.0, domains such as "chain A 42–21" were
   reported, with a 4.55 Å fit, on a structure whose chains are individually
   rigid.)
2. **Adjacent segments that still fit as one rigid body are merged.** Hinges
   are found in the initial whole-structure frame, which is a compromise fit,
   so a chain that moved rigidly still shows a deviation ramp and picks up
   spurious splits. A split is only real if the two sides cannot be fitted
   together. A hinged haemoglobin now gives 4 rigid domains at 0.000 Å
   instead of 9 with a 4.55 Å artefact.

This is a hinge *detector*, not a published flexible-alignment method; it does
not search over domain decompositions the way FATCAT or DynDom do.

---

## 7. Interpretation layer

`AlignmentResult.quality` turns the numbers into a band
(excellent/good/moderate/poor), a one-line verdict, a confidence
(high/medium/low), flagged regions and warnings. It is a pure function of
values already on the result — no I/O, no structures — and every threshold is
a named constant in `pdb_align/interpretation.py` so the whole policy is
reviewable in one place.

Bands come from TM-score when available (> 0.9 / > 0.5 / > 0.3), else RMSD
(< 1 / < 2.5 / < 5 Å). Confidence starts high and drops for coverage below
50%, for the two candidate strategies disagreeing by more than 1 Å, and for a
TM-score p-value above 0.05; a chain correspondence resting on chance-level
identity forces it to low.

Regions are flagged where the per-residue deviation exceeds
`max(2 Å, 2 × median)` in a contiguous run, broken at chain boundaries.

---

## 8. Reproducibility

- `AlignmentResult.export_bundle()` writes the aligned structure, the
  reference, the per-residue CSV, both plots, ready-to-run PyMOL (`.pml`) and
  ChimeraX (`.cxc`) scripts, and the report in text and JSON — everything
  needed to reproduce a figure.
- `AlignmentResult.save()` / `.load()` round-trip through a versioned `.npz`
  that carries no gemmi dependency, so a saved result can be re-reported and
  re-plotted later without the original files.
- `report(fmt="json")` / `to_dict()` carry every number in this document,
  including what each was normalized by.
- numba is optional and changes no result (asserted in
  `tests/test_optional_numba.py`); parallel and serial execution produce
  identical numbers (asserted in `tests/test_parallel.py`).
