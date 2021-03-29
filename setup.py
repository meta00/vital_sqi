#!/usr/bin/python3

from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='vital_sqi',
    version='0.0.1',
    description="Signal quality control pipeline for electrocardiogram and photoplethysmogram",
    long_description=long_description,
    author='Hai Ho, Khoa Le', # alphabetical order
    author_email='haihb@oucru.org, khoaldv@oucru.org',
    py_modules=['common', 'data', 'preprocess', 'sqi'],
    install_requires=['numpy',
                      'matplotlib',
                      'scipy',
                      'sklearn',
                      'pandas',
                      'tqdm',
                      'plotly',
                      'dtw-python',
                      'py-ecg-detectors'],
    zip_safe=False,
    url='https://github.com/meta00/vital_sqi',
    license='GPL 3.0',
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
)