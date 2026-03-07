import re

with open("pdb_align/core.py", "r") as f:
    text = f.read()

# Make sure the selector works with gemmi chain names
# In `_parse_chain_selector`, it returns `chain.name`. But we need to make sure we're querying correctly.
# In `def _resolve_selectors`, we need to see what `sel` is.
