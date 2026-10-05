import json
import os
import subprocess
import sys

from pdb_align import PDBAligner
from pdb_align.interpretation import AlignmentQuality

DATA = os.path.join(os.path.dirname(__file__), "data")


def _result():
    al = PDBAligner()
    al.add_reference(os.path.join(DATA, "ref.pdb"))
    al.add_mobile(os.path.join(DATA, "mob.pdb"))
    return al.align(mode="auto")


def test_quality_property_returns_assessment():
    q = _result().quality
    assert isinstance(q, AlignmentQuality)
    assert q.band in ("excellent", "good", "moderate", "poor")


def test_report_text_contains_quality_block():
    txt = _result().report(fmt="text")
    assert "Quality" in txt and "Verdict" in txt


def test_to_dict_contains_quality():
    d = _result().to_dict()
    assert "quality" in d and "band" in d["quality"]


def test_cli_text_output_shows_quality():
    out = subprocess.run(
        [sys.executable, "-m", "pdb_align",
         os.path.join(DATA, "ref.pdb"), os.path.join(DATA, "mob.pdb")],
        capture_output=True, text=True)
    assert out.returncode == 0
    assert "Verdict" in out.stdout


def test_cli_json_contains_quality():
    out = subprocess.run(
        [sys.executable, "-m", "pdb_align",
         os.path.join(DATA, "ref.pdb"), os.path.join(DATA, "mob.pdb"), "--json"],
        capture_output=True, text=True)
    assert out.returncode == 0
    payload = json.loads(out.stdout)
    assert "quality" in payload and "band" in payload["quality"]
