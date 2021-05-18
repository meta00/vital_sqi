"""
Sample 01
====================

This examples computes the three most basic standard SQIs
(kurtosis, skewness and entropy). In addition, it shows
the code in a one big section. It is just an example.

.. warning:: The ``dtw`` module seems to be loaded even though
             we are not really using it. Instead, load it
             only when required.

"""

# Libraries generic
import numpy as np
import pandas as pd

# Libraries specific
from vital_sqi.sqi.standard_sqi import perfusion_sqi
from vital_sqi.sqi.standard_sqi import kurtosis_sqi
from vital_sqi.sqi.standard_sqi import skewness_sqi
from vital_sqi.sqi.standard_sqi import entropy_sqi
from vital_sqi.sqi.standard_sqi import signal_to_noise_sqi

# ---------------------------
# Main
# ---------------------------
# Create samples
x = np.linspace(-10*np.pi, 10*np.pi, 201)

# Create signal
signal = np.sin(x)

# Loop computing sqis
k = kurtosis_sqi(signal)
s = skewness_sqi(signal)
e = entropy_sqi(signal)

# Create Series
result = pd.Series(
    data=[k, s, e],
    index=['kurtosis', 'skewness', 'entropy']
)

# Show
print("\nResult:")
print(result)

# ---------------
# Create plot
# ---------------
# Library
import matplotlib.pyplot as plt

# Plot
plt.plot(x, signal)

# Show
plt.show()