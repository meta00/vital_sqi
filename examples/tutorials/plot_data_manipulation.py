"""
Data manipulation
====================

.. note:: This is a copy of the jupyter notebook with the
          following name: Data_manipulation_ECG_PPG.ipynb.
          The other option is to use the sphinx extension
          sphinx-nbexamples.

"""

# Import generic
import os

# Import vital_sqi
from vital_sqi.data.signal_io import ECG_reader

# ------------------------------------
# Load data
# ------------------------------------
# Create path
folder_name = '../../tests/test_data'
file_name = 'example.edf'
path = os.path.join(folder_name, file_name)

# Read (there is an error when loading)
#df, info = ECG_reader(path,'edf')

# Show
#print(df)

# -----------------------------------
#
# -----------------------------------