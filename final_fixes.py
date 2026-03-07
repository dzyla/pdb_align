import re

with open("pdb_align/aligner.py", "r") as f:
    text = f.read()

text = text.replace("if not self._chosen:\n            raise ValueError(\"No alignment results available. Run align() first.\")\n\n        chosen = self._chosen",
"if not self.last_result:\n            raise ValueError(\"No alignment results available. Run align() first.\")\n\n        chosen = self.last_result[\"chosen\"]")

with open("pdb_align/aligner.py", "w") as f:
    f.write(text)

with open("pdb_align/core.py", "r") as f:
    text = f.read()

# Fix sequence-guided "mob" chain name hardcoding
text = text.replace("PseudoAtom(np.array(atA.pos.tolist(), dtype=float), atA.name, \"ref\", r_match.seqid.num, r_match.seqid.icode if hasattr(r_match.seqid, 'has_icode') and r_match.seqid.has_icode() else \"\")",
"PseudoAtom(np.array(atA.pos.tolist(), dtype=float), atA.name, r_parent_chain, r_match.seqid.num, r_match.seqid.icode if hasattr(r_match.seqid, 'has_icode') and r_match.seqid.has_icode() else \"\")")
text = text.replace("PseudoAtom(np.array(atB.pos.tolist(), dtype=float), atB.name, \"mob\", m_match.seqid.num, m_match.seqid.icode if hasattr(m_match.seqid, 'has_icode') and m_match.seqid.has_icode() else \"\")",
"PseudoAtom(np.array(atB.pos.tolist(), dtype=float), atB.name, m_parent_chain, m_match.seqid.num, m_match.seqid.icode if hasattr(m_match.seqid, 'has_icode') and m_match.seqid.has_icode() else \"\")")

with open("pdb_align/core.py", "w") as f:
    f.write(text)
