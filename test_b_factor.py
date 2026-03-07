from pdb_align.aligner import PDBAligner
aligner = PDBAligner(verbose=True)
aligner.add_reference("pdb:8UUP", chains=["A:10-150", "B"])
aligner.add_mobile("af:P00533", chains=["A"])
try:
    res = aligner.align(mode="auto", min_plddt=70)
    print("Alignment completed. RMSD:", res.rmsd)
except Exception as e:
    import traceback
    traceback.print_exc()
