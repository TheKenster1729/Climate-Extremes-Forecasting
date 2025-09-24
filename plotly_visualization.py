import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px

state = "MO"
month = "Jan"
datasets = ["era5", "merra2"]

# Load the data
max_regression = pd.read_csv(r"Regression Results/pooled_bootstrap_results_t2mmax.csv")
min_regression = pd.read_csv(r"Regression Results/pooled_bootstrap_results_t2mmin.csv")
mean_regression = pd.read_csv(r"Regression Results/pooled_bootstrap_results_t2mmean.csv")

df = max_regression
# sort by month
df = df.sort_values(by="Month", key = lambda x: x.map({"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}))
zero_point = abs(min(df["Pooled_Slope"]))/(max(df["Pooled_Slope"]) - min(df["Pooled_Slope"]))
color_scale = [(0, "#053061"), (zero_point, "white"), (1, "maroon")]

fig = px.choropleth(df, 
                    locations="Region", 
                    locationmode="USA-states", 
                    color="Pooled_Slope", 
                    scope="usa",
                    title="Maximum Temperature",
                    facet_col="Month",
                    facet_col_wrap=4,
                    color_continuous_scale=color_scale,
)
fig.update_layout(
    width=1000,
    height=500
)

fig.write_image("max_pooled_slopes.svg")