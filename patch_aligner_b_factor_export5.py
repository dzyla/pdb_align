import re

with open("pdb_align/core.py", "r") as f:
    text = f.read()

text = text.replace("p = a.get_parent()", "")
text = text.replace("chain = p.get_parent().id", "chain = a.chain_name")
text = text.replace("rid = p.get_id()", "")
text = text.replace("lbl = f\"{chain}:{rid[1]}{rid[2].strip()}\" if str(rid[2]).strip() else f\"{chain}:{rid[1]}\"",
"lbl = f\"{chain}:{a.res_seq}{a.res_icode.strip()}\" if str(a.res_icode).strip() else f\"{chain}:{a.res_seq}\"")

with open("pdb_align/core.py", "w") as f:
    f.write(text)
