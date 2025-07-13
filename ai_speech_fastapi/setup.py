from setuptools import setup, find_packages

setup(
    name='gector',
    version='0.1',
    packages=find_packages(where='gector/src'),
    package_dir={'': 'gector/src'},
)
