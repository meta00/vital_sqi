"""
Resampling
==========

This example ...

.. warning:: It does not use any vital_sqi function.

"""

# General
from scipy import signal
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------
# Load data
# -----------------------
# Create samples
x = np.linspace(-10*np.pi, 10*np.pi, 201)

# Create signal
s = np.sin(30*x)

# -----------------------
# Apply
# -----------------------
# Apply
expanded = signal.resample(s, int(len(s) * 2))
squeezed = signal.resample(s, int(len(s) / 2))

# -----------------------
# Visualize
# -----------------------
# Create figure
fig = make_subplots(rows=3, cols=1)

# Add traces
fig.add_trace(go.Scatter(x=x, y=s, name='original'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=expanded, name='expanded'), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=squeezed, name='squeezed'), row=3, col=1)

# Update layout
fig.update_layout(#height=600,
                  #width=600,
                  #title_text="Expanded/Squeezed",
                  legend=dict(
                        orientation="h",
                        yanchor="bottom", y=1.02,
                        xanchor="right", x=1)
                  )

# Show
fig
