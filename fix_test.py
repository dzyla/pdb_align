import os
with open('test_align.py', 'r') as f:
    data = f.read()

data = data.replace("aligner.get_rmsd()", "res.rmsd")
data = data.replace("aligner.get_sequence_alignment_fasta()", "res.get_sequence_alignment_fasta()")
data = data.replace("aligner.get_log()", "res.get_log()")
data = data.replace("aligner.plot_rmsd", "res.plot_rmsd")
data = data.replace("aligner.report_peaks", "res.report_peaks")
data = data.replace("aligner.save_rmsd_csv", "res.save_rmsd_csv")

with open('test_align.py', 'w') as f:
    f.write(data)
