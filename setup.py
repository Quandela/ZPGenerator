from setuptools import setup
from os.path import dirname
from glob import glob

# Package list is autogenerated to be any 'zpg' subfolder containing a __init__.py file
package_list = [dirname(p).replace('\\', '.') for p in glob('zpgenerator/**/__init__.py', recursive=True)]

setup(
    name='zpgenerator',
    author="quandela",
    version='0.2.0',
    packages=package_list,
    install_requires=[
        'qutip==4.7.3',
        'numpy',
        'scipy==1.11.4',
        'frozendict',
        'matplotlib==3.7.0'
    ],
    extras_require={
        'interactive': ['jupyter'],
    }
)