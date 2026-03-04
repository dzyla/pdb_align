from setuptools import setup, find_packages

setup(
    name="pdb_align",
    version="0.1.0",
    description="Object-oriented protein structure alignment package.",
    packages=find_packages(),
    install_requires=[
        "numpy<2.0",
        "pandas",
        "scikit-learn",
        "scipy",
        "biopython",
        "py3Dmol",
        "stmol",
        "plotly",
        "ipython_genutils",
        "matplotlib",
        "seaborn",
        "streamlit",
        "streamlit-molstar"
    ],
)
