# FILE: clip-sae-interpret/clip-sae-interpret_clean/setup.py

from setuptools import setup, find_packages

setup(
    name="sae_clip",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[],
)
