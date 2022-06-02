"""
Tappering
=========

This example ....

"""

# General
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import scipy.signal.windows as wd

# vitalSQI
from vital_sqi import preprocess

# -----------------------
# Load data
# -----------------------
# Create samples
x = np.linspace(-10*np.pi, 10*np.pi, 201)

# Create signal
s = np.sin(30*x)

# Initialize hann windows
w = list(wd.hann(len(s)))

# -----------------------
# Apply
# -----------------------
# The tapering data will pin the first and last part at the zero pivot.
# The remaining will be scale according to the windows format. The default
# tapering method shifts the segment by the value equal to the minimum value
# to the zero baseline set shift_min_to_zero=False
tap_zerobaseline_f = preprocess.taper_signal(s, shift_min_to_zero=False)

# Taper data into the zerobaseline to remove the edge effect
tap_zerobaseline_t = preprocess.taper_signal(s, shift_min_to_zero=True)

# Different windows format can be used to perform tapering process
# window is imported from the scipy package (scipy.signal.window). Default
# is using Tukey window
tap_zerobaseline_w = preprocess.taper_signal(s, window=w, shift_min_to_zero=False)

# -----------------------
# Visualize
# -----------------------
# Create figure
fig = make_subplots(rows=4, cols=1,
    subplot_titles=('Original',
                    'w=Tukey, shift=False',
                    'w=Tukey, shift=True'
                    'w=Hann, shift=False'))

# Add traces
fig.add_trace(go.Scatter(x=x, y=s, name='original'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=tap_zerobaseline_t, name='w=Tukey, shift=False'), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=tap_zerobaseline_f, name='w=Tukey, shift=True'), row=3, col=1)
fig.add_trace(go.Scatter(x=x, y=tap_zerobaseline_w, name='w=Hann, shift=False'), row=4, col=1)

# Update layout
fig.update_layout(#height=600,
                  #width=600,
                  #title_text="Smoothing",
                  legend=dict(
                      orientation="h",
                      yanchor="bottom", y=1.02,
                      xanchor="right", x=1)
                  )

# Show
fig