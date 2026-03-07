import re

with open("pdb_align/core.py", "r") as f:
    text = f.read()

text = text.replace("list(ref_struct.get_models())[0]", "ref_struct[0]")
text = text.replace("list(mob_struct.get_models())[0]", "mob_struct[0]")
text = text.replace("list(struct.get_models())[0]", "struct[0]")

with open("pdb_align/core.py", "w") as f:
    f.write(text)
