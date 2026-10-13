import os
import subprocess

import setuptools

__version__ = "2.0.0"

with open("README.md", "r") as readme:
    readme = readme.read()

with open("requirements.txt", "r") as requirements:
    requirements = requirements.read()

setuptools.setup(
    name="after",
    version=__version__,  # type: ignore
    author="Nils Demerlé",
    author_email="demerle@ircam.fr",
    description="AFTER: Audio Features Transfer and Exploration in Real-time",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={
        'after/diffusion/configs': ['*.gin'],
        'after/autoencoder/configs': ['*.gin'],
        'after/dafter/configs': ['*.gin'],
        'after/encoder_ssl/configs': ['*.gin'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "after = after_scripts.main_cli:main",
        ]
    },
    extras_require={
        "sage": [
            "einops>=0.8.1", "hydra-core>=1.3.2,<1.4", "numpy>=1.26",
            "omegaconf>=2.3.0,<2.4", "pyyaml>=6.0", "rich>=14.0",
            "soundfile>=0.13.1", "timm>=1.0.0", "huggingface-hub>=0.24,<1",
        ],
    },
    install_requires=requirements.split("\n"),
    python_requires='>=3.9',
    include_package_data=True,
)
