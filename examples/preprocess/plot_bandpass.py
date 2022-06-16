"""
Filtering (Band pass)
=====================

This example....

"""

# General
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# vitalSQI
from vital_sqi.common.band_filter import BandpassFilter


# -----------------------
# Load data
# -----------------------
# Create samples
x = np.linspace(-10*np.pi, 10*np.pi, 201)

# Create s
s = np.sin(30*x)

# -----------------------
# Apply band pass filter
# -----------------------
# Create instances
butter_bandpass = BandpassFilter("butter", fs=256)
cheby_bandpass = BandpassFilter("cheby1", fs=256)
ellip_bandpass = BandpassFilter("ellip", fs=256)

# Apply
b1 = butter_bandpass.signal_highpass_filter(s, cutoff=1, order=5)
b2 = butter_bandpass.signal_highpass_filter(s, cutoff=0.8, order=5)
b3 = butter_bandpass.signal_highpass_filter(s, cutoff=0.6, order=5)
c1 = cheby_bandpass.signal_highpass_filter(s, cutoff=1, order=5)
e1 = ellip_bandpass.signal_highpass_filter(s, cutoff=1, order=5)

# ----------------
# Visualize
# ----------------
# Create figure
fig = make_subplots(rows=7, cols=1,
    subplot_titles=('Original',
                    'f=Butter, cutoff 1Hz',
                    'f=Butter, cutoff 0.8Hz',
                    'f=Butter, cutoff 0.6Hz',
                    'f=Butter, cutoff 1Hz',
                    'f=Cheby1, cutoff 1Hz',
                    'f=Ellip, cutoff 1Hz'))

# Add traces
fig.add_trace(go.Scatter(x=x, y=s, name='original'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=b1, name='f=Butter, cutoff 1Hz'), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=b2, name='f=Butter, cutoff 0.8Hz'), row=3, col=1)
fig.add_trace(go.Scatter(x=x, y=b3, name='f=Butter, cutoff 0.6Hz'), row=4, col=1)

fig.add_trace(go.Scatter(x=x, y=b1, name='f=Butter, cutoff 1Hz'), row=5, col=1)
fig.add_trace(go.Scatter(x=x, y=c1, name='f=Cheby1, cutoff 1Hz'), row=6, col=1)
fig.add_trace(go.Scatter(x=x, y=e1, name='f=Ellip, cutoff 1Hz'), row=7, col=1)

# Update layout
fig.update_layout(height=1000,
                  #width=600,
                  #title_text="Filtering (band pass)",
                  legend=dict(
                      orientation="h",
                      yanchor="bottom", y=1.02,
                      xanchor="right", x=1)
                  )

# Show
fig