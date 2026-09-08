from setuptools import setup, find_packages
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Distribution containing platform-specific package data."""

    def has_ext_modules(self):
        return True


setup(
    name="data_graph",
    version="1.0.0",
    packages=find_packages(),
    package_data={
        'data_graph': ['*.so', '*.cpp'],
    },
    distclass=BinaryDistribution,
    install_requires=[
        'scikit-network',
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
