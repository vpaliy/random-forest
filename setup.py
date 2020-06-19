# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import sys
import re
import io

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


name = 'randforest'
description = 'Implementation of the Random Forest algorithm'
url = 'https://github.com/vpaliy/random-forest'
email = 'vpaliy97@gmail.com'
author = 'Vasyl Paliy'
requires_python = '>=2.7'
license = 'MIT'
version = None

try:
  with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = str()

try:
  with io.open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as fp:
    requires = [r.strip() for r in fp.readlines()]
except FileNotFoundError:
    requires = [
      'six',
      'future-strings'
    ]

setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=author,
    author_email=email,
    url=url,
    license=license,
    python_requires=requires_python,
    packages=find_packages(exclude=('tests',)),
    install_requires=requires,
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ]
)
