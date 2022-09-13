#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="hear-savi",
    description="An HEAR API for SAVI AudioCNN",
    version='0.0.4',
    author="Sivan",
    author_email="siwen.d@columbia.edu",
    url="https://github.com/marl/hear-semav-embedding",
    license="LICENSE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/marl/hear-semav-embedding/issues",
        "Source Code": "https://github.com/marl/hear-semav-embedding",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    install_requires=[
        "librosa",
        "typing",
        "scikit-image",
        "numba",
        "numpy",
        "torch",
        "torchaudio==0.11.0"
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
        "dev": [
            "pre-commit",
            "black",  # Used in pre-commit hooks
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
    },
)