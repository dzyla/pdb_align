"""numba is an accelerator, never a correctness dependency.

It is the dependency most likely to block an install on a fresh Python or
NumPy release, so the package must produce identical numbers without it. The
check runs in a subprocess with numba blocked at import, because un-importing
it inside the test process is not reliable.
"""
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import importlib.abc, json, sys, warnings

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "numba" or fullname.startswith("numba."):
                raise ImportError("numba blocked")
            return None

    if {block}:
        sys.meta_path.insert(0, Blocker())

    warnings.simplefilter("ignore")
    import pdb_align
    from pdb_align.core import numba_available

    out = {{"numba": numba_available()}}
    for mode in ("seq_free_shape", "seq_free_window", "seq_guided"):
        res = pdb_align.align("tests/data/1ubq.pdb", "tests/data/1crn.pdb",
                              mode=mode)
        out[mode] = [round(float(res.rmsd), 9), len(res.get_rmsd_df())]
    print(json.dumps(out))
    """
)


def _run(block):
    proc = subprocess.run([sys.executable, "-c", SCRIPT.format(block=block)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    import json
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_results_are_identical_with_and_without_numba():
    without = _run(True)
    with_numba = _run(False)
    assert without["numba"] is False
    assert without.pop("numba") is False
    with_numba.pop("numba")
    assert without == with_numba


def test_the_package_imports_without_numba():
    assert _run(True)["numba"] is False


def test_importing_pdb_align_does_not_import_numba():
    """Deferring the import keeps ~90 ms off every CLI invocation that never
    reaches the distance-matrix kernels."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, pdb_align; print('numba' in sys.modules)"],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"
