import argparse
import sys

from .aligner import PDBAligner, AlignmentFailedError


def build_parser():
    p = argparse.ArgumentParser(
        prog="pdb_align",
        description="Compare two protein structures. Prints stats; writes files only when asked.")
    p.add_argument("ref", nargs="?", help="Reference structure (file, pdb:XXXX, af:UniProtID)")
    p.add_argument("mob", nargs="?", help="Mobile structure (file, pdb:XXXX, af:UniProtID)")
    p.add_argument("--ref", dest="ref_flag", help="Reference (alias for positional)")
    p.add_argument("--mob", dest="mob_flag", help="Mobile (alias for positional)")
    p.add_argument("--ref-chains", "--ref_chains", dest="ref_chains",
                   help="Reference chains, e.g. 'A' or 'A:10-150 B'")
    p.add_argument("--mob-chains", "--mob_chains", dest="mob_chains",
                   help="Mobile chains, e.g. 'A'")
    p.add_argument("--mode", default="auto",
                   help="auto | seq_guided | seq_free_shape | seq_free_window | flexible")
    p.add_argument("--strategy", default="auto", choices=["auto", "global", "local"],
                   help="Multi-chain superposition strategy (default auto)")
    p.add_argument("--atoms", default="CA", help="CA | backbone | all_heavy")
    p.add_argument("--min-plddt", "--min_plddt", dest="min_plddt", type=float, default=0.0)
    p.add_argument("-o", "--out", help="Write aligned mobile structure to this file")
    p.add_argument("--plot", nargs="?", const="rmsd.png",
                   help="Write per-residue RMSD plot (default rmsd.png)")
    p.add_argument("--summary-plot", nargs="?", const="summary.png",
                   help="Write multi-panel summary figure")
    p.add_argument("--show", action="store_true", help="Open plots in a window")
    p.add_argument("--report", help="Write the text/JSON report to this file")
    p.add_argument("--csv", help="Write per-residue RMSD table to this CSV")
    p.add_argument("--save", help="Save full result to a .npz for later replotting")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ref = args.ref or args.ref_flag
    mob = args.mob or args.mob_flag
    if not ref or not mob:
        print("error: need a reference and a mobile structure "
              "(positional REF MOB or --ref/--mob).", file=sys.stderr)
        return 2

    # Plotting must stay headless unless the caller explicitly asks to --show
    # a window. Force the Agg backend *before* matplotlib.pyplot is imported
    # anywhere in this process (the aligner's plot methods import it lazily).
    if (args.plot is not None or args.summary_plot is not None) and not args.show:
        import matplotlib
        matplotlib.use("Agg")

    ref_chains = args.ref_chains.split() if args.ref_chains else None
    mob_chains = args.mob_chains.split() if args.mob_chains else None

    aligner = PDBAligner(verbose=args.verbose)
    try:
        aligner.add_reference(ref, chains=ref_chains)
        aligner.add_mobile(mob, chains=mob_chains)
        res = aligner.align(mode=args.mode, strategy=args.strategy,
                            atoms=args.atoms, min_plddt=args.min_plddt)
    except (AlignmentFailedError, Exception) as e:
        print(f"Alignment failed: {e}", file=sys.stderr)
        return 1

    # Default: stats to terminal. Nothing written unless a flag asks.
    if args.json:
        print(res.to_json())
    else:
        print(res.report(fmt="text"))

    if args.out:
        res.save_aligned_pdb(args.out)
        if not args.json: print(f"Wrote aligned structure: {args.out}")
    if args.csv:
        res.save_rmsd_csv(args.csv)
        if not args.json: print(f"Wrote per-residue RMSD: {args.csv}")
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(res.report(fmt="json" if args.report.endswith(".json") else "text"))
        if not args.json: print(f"Wrote report: {args.report}")
    if args.save:
        res.save(args.save)
        if not args.json: print(f"Saved result: {args.save}")
    if args.plot is not None:
        res.plot_rmsd(filename=args.plot)
        if args.show:
            import matplotlib.pyplot as plt; plt.show()
        if not args.json: print(f"Wrote RMSD plot: {args.plot}")
    if args.summary_plot is not None:
        res.plot_summary(filename=args.summary_plot, show=args.show)
        if not args.json: print(f"Wrote summary plot: {args.summary_plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
