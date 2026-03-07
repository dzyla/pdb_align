import re

with open("pdb_align/aligner.py", "r") as f:
    text = f.read()

# Make sure _process_single_alignment returns only lightweight primitives, and drops heavy references explicitly.
new_process = """def _process_single_alignment(task_payload: dict):
    \"\"\"
    Stateless module-level function for multiprocessing batch jobs.
    Avoids pickling heavy objects and drops references to prevent memory leaks.
    \"\"\"
    try:
        aligner = PDBAligner(
            ref_file=task_payload["ref_file"],
            chains_ref=task_payload["chains_ref"],
            verbose=False
        )
        aligner.add_mobile(task_payload["fpath"], chains=task_payload["chains_mob"])

        kwargs = task_payload.get("kwargs", {})
        res = aligner.align(mode=task_payload["mode"], **kwargs)

        import os
        out_pdb = os.path.join(task_payload["out_dir"], f"aligned_{task_payload['fname']}")
        res.save_aligned_pdb(out_pdb)

        # Explicitly extract scalar metrics to avoid retaining heavy AlignmentResult/Structure references
        r = {"rmsd": float(res.rmsd) if res.rmsd is not None else None,
             "tm_score": float(res.tm_score) if res.tm_score is not None else None,
             "tm_pvalue": float(res.tm_pvalue) if hasattr(res, 'tm_pvalue') and res.tm_pvalue is not None else None,
             "status": "success"}

        del res
        del aligner
        return task_payload["fname"], r
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Batch alignment failed for {task_payload.get('fname', 'Unknown')}", exc_info=True)
        return task_payload["fname"], {"status": "error", "message": str(e)}"""

text = re.sub(r'def _process_single_alignment\(task_payload: dict\):.*?(?=\n\nclass PDBAligner:)', new_process, text, flags=re.DOTALL)

with open("pdb_align/aligner.py", "w") as f:
    f.write(text)
