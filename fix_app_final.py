import re

# To solve the st.session_state parsing architecture change as requested,
# I will use the PDBAligner directly and replace the direct `_extract_ca_infos` references.

with open('struct_pair_align.py', 'r') as f:
    data = f.read()
