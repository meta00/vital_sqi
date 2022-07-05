"""
Sample 02
====================

This examples computes the three most basic standard SQIs
(kurtosis, skewness and entropy). In addition, it shows
the code and explains it in different sections. This is
similar to  jupyter notebook. It is just an example.

.. warning:: The ``dtw`` module seems to be loaded even though
             we are not really using it. Instead, load it
             only when required.

"""

###########################################
# Main
# ----
#

###########################################
# First, lets import the required libraries
#

# Libraries generic
import numpy as np
import pandas as pd

# Libraries specific
from vital_sqi.sqi.standard_sqi import perfusion_sqi
from vital_sqi.sqi.standard_sqi import kurtosis_sqi
from vital_sqi.sqi.standard_sqi import skewness_sqi
from vital_sqi.sqi.standard_sqi import entropy_sqi
from vital_sqi.sqi.standard_sqi import signal_to_noise_sqi

#########################################
# Lets create the sinusoide
#

# Create samples
x = np.linspace(-10*np.pi, 10*np.pi, 201)

# Create signal
signal = np.sin(x)

#########################################
# Lets compute the SQIs
#

# Loop computing sqis
k = kurtosis_sqi(signal)
s = skewness_sqi(signal)
e = entropy_sqi(signal)

#########################################
# Lets display the result
#

# Create Series
result = pd.Series(
    data=[k, s, e],
    index=['kurtosis', 'skewness', 'entropy']
)

# Show
print("\nResult:")
print(result)

###########################################
# Plot
# ----
#
# Lets plot something so it shows a thumbicon
#

# ---------------
# Create plot
# ---------------
# Library
import matplotlib.pyplot as plt

# Plot
plt.plot(x, signal)

# Show
plt.show()