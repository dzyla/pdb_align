import argparse
import sys
from .aligner import PDBAligner

def main():
    parser = argparse.ArgumentParser(description="pdb_align: 3D structural protein alignment tool")
    parser.add_argument("--ref", required=True, help="Reference structure (file path or pdb:XXXX or af:XXXX)")
    parser.add_argument("--ref_chains", help="Reference chains to align (e.g. A, or A:10-50 B)")
    parser.add_argument("--mob", required=True, help="Mobile structure (file path or pdb:XXXX or af:XXXX)")
    parser.add_argument("--mob_chains", help="Mobile chains to align (e.g. A)")
    parser.add_argument("--mode", default="auto", help="Alignment mode (auto, seq_guided, seq_free_shape, seq_free_window)")
    parser.add_argument("--out", help="Output file path for aligned structure (e.g. aligned.cif or aligned.pdb)")
    parser.add_argument("--atoms", default="CA", help="Atoms to use (CA, backbone, all_heavy)")
    parser.add_argument("--min_plddt", type=float, default=0.0, help="Minimum pLDDT/B-factor to include atoms")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    ref_chains = args.ref_chains.split() if args.ref_chains else None
    mob_chains = args.mob_chains.split() if args.mob_chains else None

    aligner = PDBAligner(verbose=args.verbose)

    try:
        aligner.add_reference(args.ref, chains=ref_chains)
        aligner.add_mobile(args.mob, chains=mob_chains)

        res = aligner.align(
            mode=args.mode,
            atoms=args.atoms,
            min_plddt=args.min_plddt
        )

        print(f"Alignment successful.")
        print(f"Chosen Method: {res._chosen['name']}")
        print(f"RMSD: {res.rmsd:.3f} Å" if res.rmsd is not None else "RMSD: None")
        print(f"TM-Score: {res.tm_score:.4f}" if res.tm_score is not None else "TM-Score: None")
        print(f"TM-Score p-value: {res.tm_pvalue:.4e}" if res.tm_pvalue is not None else "TM-Score p-value: None")

        if args.out:
            res.save_aligned_pdb(args.out)
            print(f"Saved aligned structure to {args.out}")

    except Exception as e:
        print(f"Alignment failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
