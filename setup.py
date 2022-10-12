#!/usr/bin/python3

from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name = 'vital_sqi',
    version = '0.1.0',
    packages = find_packages(include = ["vital_sqi", "vital_sqi.*"]),
    description = "Signal quality control pipeline for electrocardiogram and "
                 "photoplethysmogram",
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = 'Khoa Le, Hai Ho, Stefan Karolcik, Heloise Greeff',
    author_email = 'khoaldv@oucru.org, haihb@oucru.org, s.karolcik@imperial.ac.uk, '
                   'heloise.greeff@eng.ox.ac.uk',
    maintainer = 'Hai Ho, Khoa Le',
    maintainer_email = 'haihb@oucru.org, khoaldv@oucru.org',
    py_modules = ['common', 'data', 'preprocess', 'sqi'],
    install_requires = ['librosa>=0.9.2',
                        'heartpy>=1.2.6',
                        'pmdarima>=1.8.0',
                        'hrv-analysis>=1.0.3',
                        'matplotlib>=3.3.3',
                        'numpy>=1.20.2',
                        'pandas>=1.1.5',
                        'plotly>=4.14.3',
                        'scikit-learn>=0.24.1',
                        'scipy>=1.6.0',
                        'statsmodels>=0.12.1',
                        'tqdm>=4.56.0',
                        'py-ecg-detectors>=1.0.2',
                        'pyEDFlib>=0.1.20',
                        'pycwt>=0.3.0a22',
                        'wfdb>=3.3.0',
                        'datetimerange>=1.0.0',
                        'dateparser>=1.0.0',
                        'openpyxl>=3.0.7',
                        'pyflowchart>=0.1.3'],
    python_requires = '>=3.7',
    zip_safe = False,
    url = 'https://github.com/meta00/vital_sqi',
    license = 'MIT',
    classifiers = [
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
)
