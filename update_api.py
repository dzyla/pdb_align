import re

with open("pdb_align/aligner.py", "r") as f:
    data = f.read()

# I will write a script to extract the logic out of PDBAligner and build the AlignmentResult class.
# Because this file is getting huge, it might be better to do this in python.
