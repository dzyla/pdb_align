"""Parallel execution must be an optimisation, not a second code path.

Running over a process pool has to produce the same numbers as running in
process, and a model that cannot be scored has to stay visible — a warning
raised inside a worker never reaches the parent, so the errors are carried
back and re-emitted.
"""
import glob
import os
import warnings

import gemmi
import numpy as np
import pytest

import pdb_align
from pdb_align import evaluate_models
from pdb_align.aligner import _resolve_workers

REF = "tests/data/4hhb_bb.pdb"
N_MODELS = 4


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """A handful of perturbed copies of the reference complex."""
    out = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(0)
    paths = []
    for k in range(N_MODELS):
        st = gemmi.read_structure(REF)
        st.setup_entities()
        for chain in st[0]:
            for res in chain:
                for atom in res:
                    v = np.array(atom.pos.tolist()) + rng.normal(scale=0.3 + 0.2 * k,
                                                                 size=3)
                    atom.pos = gemmi.Position(*v)
        p = out / f"m{k}.pdb"
        st.write_pdb(str(p))
        paths.append(str(p))
    return paths


def _summary_values(table):
    cols = [c for c in table.columns if c != "capri"]
    return table.sort_values("model")[cols].round(9).to_dict(orient="records")


def test_evaluate_models_is_identical_serial_and_parallel(models):
    serial = evaluate_models(REF, models, receptor_chains=["A", "B"],
                             ligand_chains=["C", "D"], workers=1)
    parallel = evaluate_models(REF, models, receptor_chains=["A", "B"],
                               ligand_chains=["C", "D"], workers=2)
    assert _summary_values(serial.table) == _summary_values(parallel.table)
    assert serial.best == parallel.best


def test_align_ensemble_is_identical_serial_and_parallel(models):
    al = pdb_align.PDBAligner()
    al.add_reference(REF)
    serial = al.align_ensemble(models, workers=1).summary()
    al2 = pdb_align.PDBAligner()
    al2.add_reference(REF)
    parallel = al2.align_ensemble(models, workers=2).summary()
    assert serial.round(9).to_dict() == parallel.round(9).to_dict()


def test_parallel_ensemble_results_support_the_analysis_methods(models):
    """Worker results cross a process boundary as plain data, so the ensemble
    analyses must still work on them (a gemmi.Structure cannot be pickled)."""
    al = pdb_align.PDBAligner()
    al.add_reference(REF)
    ens = al.align_ensemble(models, workers=2)
    assert len(ens.results) == len(models)
    assert ens.rmsd_matrix().shape == (len(models), len(models))
    assert len(ens.cluster(n_clusters=2)) == len(models)
    assert "RMSD" in ens.results[0].get_rmsd_df().columns
    assert ens.results[0].quality.band


def test_a_failing_model_is_reported_not_silently_dropped(models, tmp_path):
    """A model that cannot be read must warn and leave NaNs, in both modes."""
    broken = tmp_path / "broken.pdb"
    broken.write_text("this is not a structure\n")
    with pytest.warns(UserWarning):
        ev = evaluate_models(REF, list(models) + [str(broken)], workers=2)
    assert len(ev.table) == len(models) + 1
    row = ev.table[ev.table["model"] == "broken"].iloc[0]
    assert np.isnan(row["tm_score"])


def test_a_failing_model_in_an_ensemble_warns_in_parallel(models, tmp_path):
    broken = tmp_path / "broken2.pdb"
    broken.write_text("not a structure\n")
    al = pdb_align.PDBAligner()
    al.add_reference(REF)
    with pytest.warns(UserWarning, match="skipping"):
        ens = al.align_ensemble(list(models) + [str(broken)], workers=2)
    assert len(ens.results) == len(models)


@pytest.mark.parametrize("requested,n_tasks,expected", [
    (1, 10, 1),
    (4, 10, 4),
    (8, 3, 3),       # never more workers than tasks
    (4, 1, 1),       # a single task stays in-process
    (-1, 2, min(2, os.cpu_count() or 1)),
])
def test_worker_count_resolution(requested, n_tasks, expected):
    assert _resolve_workers(requested, n_tasks) == expected


def test_parallel_map_falls_back_to_serial_instead_of_crashing():
    """Parallelism is an optimisation, so a pool that cannot start must not
    lose the result.

    forkserver/spawn need the calling __main__ to be importable, which it is
    not for piped code, some notebook setups, or a nested parallel call. That
    used to surface as BrokenProcessPool. A locally-defined function cannot be
    pickled, which reproduces the same class of failure.
    """
    from pdb_align.aligner import _map_parallel

    def square(x):  # not importable from a worker
        return x * x

    with pytest.warns(UserWarning, match="continuing in this process"):
        out = _map_parallel(square, [1, 2, 3, 4], 2, "test")
    assert out == [1, 4, 9, 16]


def test_parallel_map_prefers_a_thread_safe_start_method():
    """Plain fork copies numpy/BLAS thread locks and can deadlock the child, so
    it must never be the first choice."""
    from pdb_align.aligner import _START_METHODS

    assert _START_METHODS[0] == "forkserver"
    assert _START_METHODS.index("fork") == len(_START_METHODS) - 1
