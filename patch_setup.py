with open("setup.py", "r") as f:
    text = f.read()

# Replace the two setups with a single one that merges everything
merged_setup = """from setuptools import setup, find_packages

setup(
    name="pdb_align",
    version="0.1.0",
    description="Object-oriented protein structure alignment package.",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "pdb_align=pdb_align.__main__:main",
        ]
    },
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "biopython",
        "gemmi",
        "requests"
    ],
    extras_require={
        "app": [
            "py3Dmol",
            "stmol",
            "plotly",
            "ipython_genutils",
            "matplotlib",
            "seaborn",
            "streamlit",
            "streamlit-molstar"
        ]
    }
)"""

with open("setup.py", "w") as f:
    f.write(merged_setup)
