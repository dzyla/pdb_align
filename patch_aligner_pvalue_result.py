import re

with open("pdb_align/result.py", "r") as f:
    text = f.read()

new_property = """    @property
    def tm_pvalue(self) -> Optional[float]:
        \"\"\"
        Returns the p-value of the TM-score for the aligned structures.
        \"\"\"
        tm = self.tm_score
        if tm is None:
            return None

        import pdb_align.metrics as metrics
        return metrics.calculate_tm_pvalue(tm, self.ref_length)
"""

text = re.sub(
    r'    @property\n    def rotation_matrix',
    new_property + '\n    @property\n    def rotation_matrix',
    text
)

with open("pdb_align/result.py", "w") as f:
    f.write(text)
