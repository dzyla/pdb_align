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
        "numba",
        "requests"
    ],
    extras_require={
        "app": [
            "streamlit",
            "py3Dmol",
            "stmol",
            "plotly",
            "matplotlib",
            "seaborn",
            "streamlit-molstar"
        ],
        "dev": ["pytest"]
    }
)
