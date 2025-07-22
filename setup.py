from setuptools import setup, find_packages

setup(
    name="data_graph",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "numba"
    ],
    author="Connor Frankston",
    description="Tools for semimetric graph construction",
    python_requires=">=3.7",
)
