import os
from pdb_align.aligner import PDBAligner

def test_pdb_aligner():
    print("Initializing PDBAligner...")
    aligner = PDBAligner("1CRN.pdb", chains_ref=["A"])
    aligner.add_mobile("2CRN.pdb", chains=["A"])

    print("Testing auto mode...")
    try:
        res = aligner.align(mode="auto")
        print(f"Auto mode successful. RMSD: {aligner.get_rmsd()}, Chosen: {res['chosen']['name']}")
    except Exception as e:
        print(f"Auto mode error: {e}")

    print("Testing seq_free_shape mode...")
    try:
        res = aligner.align(mode="seq_free_shape")
        print(f"Shape mode successful. RMSD: {aligner.get_rmsd()}, Chosen: {res['chosen']['name']}")
    except Exception as e:
        print(f"Shape mode error: {e}")

    print("Testing seq_free_window mode...")
    try:
        res = aligner.align(mode="seq_free_window")
        print(f"Window mode successful. RMSD: {aligner.get_rmsd()}, Chosen: {res['chosen']['name']}")
    except Exception as e:
        print(f"Window mode error: {e}")

    print("Testing seq_guided mode...")
    try:
        res = aligner.align(mode="seq_guided")
        print(f"Seq guided mode successful. RMSD: {aligner.get_rmsd()}, Chosen: {res['chosen']['name']}")
    except Exception as e:
        print(f"Seq guided mode error: {e}")

    print("Testing batch alignment...")
    os.makedirs("batch_test_mob", exist_ok=True)
    import shutil
    shutil.copy("2CRN.pdb", "batch_test_mob/2CRN_copy.pdb")
    try:
        results = aligner.batch_align("batch_test_mob", "batch_test_out", mode="auto")
        print(f"Batch align successful. Results: {results}")
    except Exception as e:
        print(f"Batch align error: {e}")
    finally:
        shutil.rmtree("batch_test_mob", ignore_errors=True)
        shutil.rmtree("batch_test_out", ignore_errors=True)

if __name__ == "__main__":
    if not os.path.exists("1CRN.pdb") or not os.path.exists("2CRN.pdb"):
        print("Please run get_pdbs.py to download test PDBs.")
    else:
        test_pdb_aligner()
