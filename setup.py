# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:45:03 2021

@author: hyzha
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="astrofix", # Replace with your own username
    version="0.0.1",
    author="Hengyue Zhang, Timothy Brandt",
    author_email="hengyue@ucsb.edu, tbrandt@physics.ucsb.edu",
    description="Astronomical image correction algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HengyueZ/astrofix",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)