import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

state = "MO"
month = "Jan"
dataset = "merra2"

# Load the data
max_temps = pd.read_csv(r"full_processed_data_t2mmax.csv")
mean_temps = pd.read_csv(r"full_processed_data_t2mmean.csv")
min_temps = pd.read_csv(r"full_processed_data_t2mmin.csv")

# Filter the data
max_temp = max_temps[(max_temps["Region"] == state)
                     & (max_temps["Month"] == month)
                     & (max_temps["Dataset"] == dataset)]

mean_temp = mean_temps[(mean_temps["Region"] == state)
                     & (mean_temps["Month"] == month)
                     & (mean_temps["Dataset"] == dataset)]

min_temp = min_temps[(min_temps["Region"] == state)
                     & (min_temps["Month"] == month)
                     & (min_temps["Dataset"] == dataset)]

# Create figure with shared y-axis
fig = make_subplots(rows=1, cols=3, 
                    subplot_titles=("Minimum Temperature", "Mean Temperature", "Maximum Temperature"),
                    shared_yaxes=True)

# Add traces for each temperature type
for idx, temp_data in enumerate([min_temp, mean_temp, max_temp]):
    # Create regression line
    x = temp_data["Global_Temp"]
    y = temp_data["Average_Temperature"]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", 
                  name="Historical Data",
                  marker=dict(color="#CBC3E3"),
                  showlegend=False),
        row=1, col=idx+1
    )
    
    # Add regression line
    fig.add_trace(
        go.Scatter(x=x_line, y=p(x_line), mode="lines",
                  name="Trend Line",
                  line=dict(color="#56A3A6", width=2),
                  showlegend=False),
        row=1, col=idx+1
    )

# Update layout
fig.update_layout(
    height=400, 
    width=1200, 
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=50, r=50, t=20, b=50)
)
fig.update_xaxes(title_text="Global Temperature", showgrid=False)
fig.update_yaxes(title_text="Average Temperature", showgrid=False)

fig.show()
