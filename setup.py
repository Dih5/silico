#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script"""

from setuptools import setup, find_packages
import os

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

setup(author="Dih5",
      author_email='dihedralfive@gmail.com',
      classifiers=[
          'Development Status :: 3 - Alpha',
          # 'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
      ],
      description='Python package to handle in silico experiments',
      extras_require={
          "docs": ["nbsphinx", "sphinx-rtd-theme", "IPython"],
          "test": ["pytest"],
          "progress": ["tqdm"]
      },
      keywords=[],
      long_description=long_description,
      long_description_content_type='text/markdown',
      name='insilico',
      packages=find_packages(include=['insilico'], exclude=["demos", "tests", "docs"]),
      install_requires=["numpy", "pandas", "scipy"],
      url='https://github.com/Dih5/insilico',
      version='0.1.0',

      )
