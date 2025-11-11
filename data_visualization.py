import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

"""
=== Options ===
1. Visualize global average temperature (both ERA5 and MERRA2)
2. Visualize local monthly average temperature (choose region, month, and dataset)

=== Usage ===
For global average temperature, set plot_type = "global", and temp_var = "min", "mean", or "max"
for minimum, mean, or maximum average global temperature. Month, region, and dataset
do not matter, so can be any value.

For local monthly average temperature, set plot_type = "local", and temp_var = "min", "mean", or "max"
for minimum, mean, or maximum average local temperature. Region, month, and dataset
must be specified.
"""

plot_type = "global"
temp_var = "max"
month = "Jan"
region = "MO"
dataset = "era5"

def global_average_temperature(temp_var):
    if temp_var == "min":



if __name__ == "__main__":
    pass
