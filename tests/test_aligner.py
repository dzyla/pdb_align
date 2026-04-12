# tests/test_aligner.py
import os
import pytest
from unittest.mock import patch
import gemmi

from pdb_align.aligner import PDBAligner


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
