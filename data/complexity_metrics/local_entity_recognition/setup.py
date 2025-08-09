#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="local-entity-recognition",
    version="1.0.0",
    description="Local entity recognition system as DBpedia alternative",
    packages=find_packages(),
    install_requires=[
        "nltk",
        "spacy",
    ],
    python_requires=">=3.7",
)
