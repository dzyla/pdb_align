from pdb_align.interpretation import assess, AlignmentQuality, FlaggedRegion


def _uniform(chain, n, val):
    return [(chain, f"{chain}:{i}", val) for i in range(1, n + 1)]


def test_identical_is_excellent_high_confidence():
    q = assess(tm_score=0.99, rmsd=0.2, coverage_pct=100.0, n_aligned=100,
               per_residue=_uniform("A", 100, 0.2), chain_mapping=None,
               candidate_rmsds=[0.2, 0.2])
    assert q.band == "excellent"
    assert q.confidence == "high"
    assert q.flagged_regions == []
    assert not q.warnings


def test_flags_contiguous_high_rmsd_region():
    pr = _uniform("A", 20, 0.5)
    for k in (9, 10, 11, 12):  # residues A:10..A:13, above max(2.0, 2*median)
        pr[k] = ("A", f"A:{k+1}", 6.0)
    q = assess(tm_score=0.7, rmsd=1.5, coverage_pct=100.0, n_aligned=20,
               per_residue=pr, chain_mapping=None, candidate_rmsds=[1.5, 1.6])
    assert len(q.flagged_regions) == 1
    fr = q.flagged_regions[0]
    assert fr.chain == "A" and fr.n_residues == 4
    assert fr.max_rmsd == 6.0 and fr.kind == "deviation"


def test_hinge_region_kind():
    q = assess(tm_score=0.7, rmsd=2.0, coverage_pct=100.0, n_aligned=50,
               per_residue=_uniform("A", 50, 1.0), chain_mapping=None,
               candidate_rmsds=[2.0, 2.1],
               hinge_regions=[("A", "A:20", "A:30")])
    assert any(fr.kind == "hinge" for fr in q.flagged_regions)


def test_low_coverage_warns_and_lowers_confidence():
    q = assess(tm_score=0.6, rmsd=2.0, coverage_pct=30.0, n_aligned=30,
               per_residue=_uniform("A", 30, 2.0), chain_mapping=None,
               candidate_rmsds=[2.0, 2.1])
    assert any("coverage" in w.lower() for w in q.warnings)
    assert q.confidence in ("low", "medium")


def test_candidate_disagreement_lowers_confidence():
    q = assess(tm_score=0.6, rmsd=1.0, coverage_pct=90.0, n_aligned=90,
               per_residue=_uniform("A", 90, 1.0), chain_mapping=None,
               candidate_rmsds=[1.0, 4.0])
    assert q.confidence in ("low", "medium")


def test_band_falls_back_to_rmsd_without_tm():
    q = assess(tm_score=None, rmsd=0.5, coverage_pct=100.0, n_aligned=50,
               per_residue=_uniform("A", 50, 0.5), chain_mapping=None,
               candidate_rmsds=[0.5, 0.5])
    assert q.band == "excellent"


def test_to_dict_roundtrips():
    q = assess(tm_score=0.99, rmsd=0.2, coverage_pct=100.0, n_aligned=100,
               per_residue=_uniform("A", 100, 0.2), chain_mapping=None,
               candidate_rmsds=[0.2, 0.2])
    d = q.to_dict()
    assert d["band"] == "excellent" and "verdict" in d and "flagged_regions" in d
