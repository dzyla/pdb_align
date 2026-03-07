import re

with open("pdb_align/aligner.py", "r") as f:
    text = f.read()

# Let's fix the try except that swallows sequence free exception to print traceback so we know why it fails
text = text.replace("except Exception as e:\n                import logging\n                logging.getLogger(__name__).warning(\"Sequence-free alignment failed\", exc_info=True)",
"except Exception as e:\n                import traceback\n                traceback.print_exc()")

with open("pdb_align/aligner.py", "w") as f:
    f.write(text)
