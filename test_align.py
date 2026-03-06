import os
from pdb_align.aligner import PDBAligner

def test_pdb_aligner():
    print("Initializing PDBAligner...")
    aligner = PDBAligner("1CRN.pdb", chains_ref=["A"])
    aligner.add_mobile("2CRN.pdb", chains=["A"])

    print("Testing auto mode...")
    try:
        res = aligner.align(mode="auto")
        print(f"Auto mode successful. RMSD: {res.rmsd}, Chosen: {res._chosen['name']}")
    except Exception as e:
        print(f"Auto mode error: {e}")

    print("Testing seq_free_shape mode...")
    try:
        res = aligner.align(mode="seq_free_shape")
        print(f"Shape mode successful. RMSD: {res.rmsd}, Chosen: {res._chosen['name']}")
    except Exception as e:
        print(f"Shape mode error: {e}")

    print("Testing seq_free_window mode...")
    try:
        res = aligner.align(mode="seq_free_window")
        print(f"Window mode successful. RMSD: {res.rmsd}, Chosen: {res._chosen['name']}")
    except Exception as e:
        print(f"Window mode error: {e}")

    print("Testing seq_guided mode...")
    try:
        res = aligner.align(mode="seq_guided")
        print(f"Seq guided mode successful. RMSD: {res.rmsd}, Chosen: {res._chosen['name']}")
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

    print("Testing report_peaks...")
    try:
        peaks = res.report_peaks(on="reference", top_n=3)
        print(f"report_peaks successful: {peaks}")
    except Exception as e:
        print(f"report_peaks error: {e}")

    print("Testing plot_rmsd...")
    try:
        res.plot_rmsd("test_rmsd_plot.pdf")
        if os.path.exists("test_rmsd_plot.pdf"):
            print("plot_rmsd successful: File created.")
        else:
            print("plot_rmsd error: File not created.")
    except Exception as e:
        print(f"plot_rmsd error: {e}")

    print("Testing get_sequence_alignment_fasta...")
    try:
        aligner.align(mode="seq_guided")
        fasta = res.get_sequence_alignment_fasta()
        if ">Reference_1CRN" in fasta and ">Mobile_2CRN" in fasta:
            print("get_sequence_alignment_fasta successful.")
        else:
            print("get_sequence_alignment_fasta failed formatting. Output:")
            print(fasta)
    except Exception as e:
        print(f"get_sequence_alignment_fasta error: {e}")

    print("Testing get_log...")
    try:
        log = res.get_log()
        if "PDB Aligner Result Log" in log and "RMSD" in log:
            print("get_log successful.")
        else:
            print("get_log failed formatting.")
    except Exception as e:
        print(f"get_log error: {e}")

    print("Testing set_reference_chains and set_mobile_chains...")
    try:
        aligner.set_reference_chains(["A"])
        aligner.set_mobile_chains(["A"])
        aligner.align(mode="auto")
        print("Chain setters successful.")
    except Exception as e:
        print(f"Chain setters error: {e}")

    print("Testing save_rmsd_csv...")
    try:
        res.save_rmsd_csv("test_rmsd.csv")
        if os.path.exists("test_rmsd.csv"):
            print("save_rmsd_csv successful.")
            os.remove("test_rmsd.csv")
        else:
            print("save_rmsd_csv failed.")
    except Exception as e:
        print(f"save_rmsd_csv error: {e}")

    print("Testing general sequence alignment...")
    try:
        res = aligner.get_general_sequence_alignment("A", "A")
        if res:
            print("get_general_sequence_alignment successful.")
        else:
            print("get_general_sequence_alignment failed.")
    except Exception as e:
        print(f"get_general_sequence_alignment error: {e}")

if __name__ == "__main__":
    if not os.path.exists("1CRN.pdb") or not os.path.exists("2CRN.pdb"):
        print("Please run get_pdbs.py to download test PDBs.")
    else:
        test_pdb_aligner()
