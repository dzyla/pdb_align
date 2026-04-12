# tests/test_aligner.py
import os
import pytest
from unittest.mock import patch
import gemmi

import pdb_align
from pdb_align.aligner import PDBAligner, AlignmentResult, DomainResult


def test_structure_cache_set_reference(tmp_path):
    """Parsing the same file twice should only call gemmi.read_structure once."""
    pdb_content = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  ALA A   3       7.000   8.000   9.000  1.00  0.00           C
END
"""
    pdb_file = tmp_path / "test_ref.pdb"
    pdb_file.write_text(pdb_content)

    aligner = PDBAligner()
    aligner.set_reference(str(pdb_file))
    first_struct = aligner._struct_cache.get(str(pdb_file.resolve()))
    assert first_struct is not None

    # Setting reference again should reuse cache, not re-parse
    with patch("pdb_align.aligner._parse_path", wraps=lambda p: gemmi.read_structure(p)) as mock_parse:
        aligner.set_reference(str(pdb_file))
        mock_parse.assert_not_called()


def test_structure_cache_add_mobile(tmp_path):
    """add_mobile with the same path twice should not re-parse."""
    pdb_content = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  ALA A   3       7.000   8.000   9.000  1.00  0.00           C
END
"""
    ref = tmp_path / "ref.pdb"
    mob = tmp_path / "mob.pdb"
    ref.write_text(pdb_content)
    mob.write_text(pdb_content)

    aligner = PDBAligner()
    aligner.set_reference(str(ref))
    aligner.add_mobile(str(mob))
    assert aligner._struct_cache.get(str(mob.resolve())) is not None

    with patch("pdb_align.aligner._parse_path", wraps=lambda p: gemmi.read_structure(p)) as mock_parse:
        aligner.add_mobile(str(mob))
        mock_parse.assert_not_called()


def test_top_level_align_chains_kwargs(tmp_path):
    """pdb_align.align() should pass chains_ref/chains_mob correctly and return AlignmentResult."""
    pdb_content = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  ALA A   3       7.000   8.000   9.000  1.00  0.00           C
END
"""
    ref = tmp_path / "ref.pdb"
    mob = tmp_path / "mob.pdb"
    ref.write_text(pdb_content)
    mob.write_text(pdb_content)

    result = pdb_align.align(str(ref), str(mob))
    assert isinstance(result, AlignmentResult)
    assert result.rmsd is not None
    assert result.rmsd >= 0.0

    # verify chain filtering is forwarded correctly
    result_chain = pdb_align.align(str(ref), str(mob), chains_ref=["A"], chains_mob=["A"])
    assert isinstance(result_chain, AlignmentResult)
    assert result_chain.rmsd is not None


def test_domain_result_creation():
    """DomainResult should hold all expected fields."""
    import numpy as np
    dr = DomainResult(
        domain_id=0,
        chain_id="A",
        residue_start=1,
        residue_end=80,
        n_residues=80,
        rmsd=1.5,
        rotation=np.eye(3),
        translation=np.zeros(3),
    )
    assert dr.domain_id == 0
    assert dr.chain_id == "A"
    assert dr.n_residues == 80
    assert dr.rmsd == 1.5


def test_alignment_result_domains_none_by_default(tmp_path):
    """AlignmentResult.domains should be None when mode is not flexible."""
    pdb_content = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  ALA A   3       7.000   8.000   9.000  1.00  0.00           C
END
"""
    ref = tmp_path / "ref.pdb"
    mob = tmp_path / "mob.pdb"
    ref.write_text(pdb_content)
    mob.write_text(pdb_content)

    aligner = PDBAligner()
    aligner.add_reference(str(ref))
    aligner.add_mobile(str(mob))
    result = aligner.align(mode="auto")
    assert result.domains is None


def test_alignment_result_flexible_rmsd_weighted_average():
    """AlignmentResult.rmsd returns weighted average when domains are set."""
    import numpy as np

    chosen = {"seqguided": None, "seqfree": None, "name": "flexible", "reason": "test"}

    dr1 = DomainResult(0, "A", 1, 50, 50, rmsd=1.0, rotation=np.eye(3), translation=np.zeros(3))
    dr2 = DomainResult(1, "A", 51, 100, 50, rmsd=3.0, rotation=np.eye(3), translation=np.zeros(3))

    result = AlignmentResult(
        chosen=chosen, seqguided=None, seqfree=None,
        ref_file="ref.pdb", mob_file="mob.pdb", mob_struct=None,
        ref_lens={"A": 100}, mob_lens={"A": 100},
        domains=[dr1, dr2],
    )
    # Weighted average: (1.0*50 + 3.0*50) / 100 = 2.0
    assert result.rmsd == pytest.approx(2.0)


def test_flexible_alignment_produces_domains(tmp_path):
    """mode='flexible' should return an AlignmentResult with .domains populated."""
    import numpy as np
    import math

    # Reference: 60-residue straight chain along X
    ref_lines = []
    atom_num = 1
    for i in range(60):
        x = float(i) * 3.8
        ref_lines.append(
            f"ATOM  {atom_num:5d}  CA  ALA A{i+1:4d}    {x:8.3f}   0.000   0.000  1.00  0.00           C"
        )
        atom_num += 1
    ref_lines.append("END")

    # Mobile: first 30 residues same as ref; last 30 rotated 45° in XY plane
    mob_lines = []
    atom_num = 1
    for i in range(30):
        x = float(i) * 3.8
        mob_lines.append(
            f"ATOM  {atom_num:5d}  CA  ALA A{i+1:4d}    {x:8.3f}   0.000   0.000  1.00  0.00           C"
        )
        atom_num += 1
    angle = math.radians(45)
    for i in range(30):
        x_orig = float(i + 30) * 3.8
        x = x_orig * math.cos(angle)
        y = x_orig * math.sin(angle)
        mob_lines.append(
            f"ATOM  {atom_num:5d}  CA  ALA A{i+31:4d}    {x:8.3f} {y:8.3f}   0.000  1.00  0.00           C"
        )
        atom_num += 1
    mob_lines.append("END")

    ref = tmp_path / "ref.pdb"
    mob = tmp_path / "mob.pdb"
    ref.write_text("\n".join(ref_lines))
    mob.write_text("\n".join(mob_lines))

    aligner = PDBAligner()
    aligner.add_reference(str(ref))
    aligner.add_mobile(str(mob))
    result = aligner.align(mode="flexible", hinge_threshold=2.0, domain_min_residues=10)

    assert result.domains is not None
    assert len(result.domains) >= 1
    for dr in result.domains:
        assert dr.rmsd >= 0.0
        assert dr.n_residues > 0
    assert result.rmsd is not None
    assert result.rmsd >= 0.0
