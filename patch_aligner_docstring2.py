import re

with open("pdb_align/aligner.py", "r") as f:
    text = f.read()

new_docstring = """def get_tm_score(self, normalize_by: str = 'reference') -> Optional[float]:
        \"\"\"
        Calculates the TM-score.

        normalize_by: 'reference', 'mobile', or 'min'.

        Warning: Standard TM-align always normalizes by the length of the *reference*
        protein. Changing the normalization length breaks TM-score comparability
        across different targets.
        \"\"\""""

text = re.sub(
    r'def get_tm_score\(self, normalize_by: str = \'reference\'\) -> Optional\[float\]:\n\s+"""\n\s+Calculates the TM-score\.\n\n\s+normalize_by: \'reference\', \'mobile\', or \'min\'\n\s+"""',
    new_docstring,
    text
)

with open("pdb_align/aligner.py", "w") as f:
    f.write(text)
