"""
Smoothing
=========

This example shows how to apply convolutional window to smooth
the signal (the default windows is flat and can be assigned
with different distribution)

"""

# General
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# vitalSQI
from vital_sqi.preprocess.preprocess_signal import smooth_signal
from vital_sqi.common.utils import generate_timestamp

# -----------------------
# Load data
# -----------------------
# Create samples
x = np.linspace(-10*np.pi, 10*np.pi, 201)

# Create signal
signal = np.sin(30*x)
# timestamps = generate_timestamp(None,100,len(s))
# signal = pd.DataFrame(np.hstack((np.array(timestamps).reshape(-1,1),
#                                  np.array(s).reshape(-1,1))))


# -----------------------
# Apply
# -----------------------
# Smoothing
smooth1 = smooth_signal(signal)
smooth2 = smooth_signal(signal, window_len=9)

# -----------------------
# Visualize
# -----------------------
# Create figure
fig = make_subplots(rows=3, cols=1)

# Add traces
fig.add_trace(go.Scatter(x=x, y=signal, name='original'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=smooth1, name='smoothing, length=5'), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=smooth2, name='smoothing, length=9'), row=3, col=1)

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
fig.show()
