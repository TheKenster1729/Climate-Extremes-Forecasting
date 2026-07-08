import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

"""
=== Options ===
1. Visualize global average temperature (both ERA5 and MERRA2)
2. Visualize local monthly average temperature (choose region, month, and dataset)

=== Usage ===
For global average temperature, set plot_type = "global". The global mean average temperature
was used as the predictor variable for all regressions, so global max and min were not calculated.

For local monthly average temperature, set plot_type = "local", and temp_var = "min", "mean", or "max"
for minimum, mean, or maximum average local temperature. Region must be specified.
"""

plot_type = "global"
temp_var = "min"
region = "AL"

def global_average_temperature():
    df = pd.read_csv(f"full_processed_data_t2mmean.csv")
    df = df[(df["Month"] == "Jan") & (df["Region"] == "MA")]
    fig = px.line(df, x="Year", y="Global_Temp", title=f"Global {temp_var.capitalize()} Temperature", color="Dataset", color_discrete_map={"era5": "#33b1ff", "merra2": "#24a148"})
    fig.update_layout(legend_title_text="Dataset")
    fig.for_each_yaxis(lambda yaxis: yaxis.update(title_text=None))
    fig.update_yaxes(title_text=f"Global Average Temperature (°C)", row = 2, col = 1)
    fig.for_each_trace(lambda t: t.update(name=t.name.upper()))

    return fig

def local_average_temperature(temp_var, region):
    df = pd.read_csv(f"full_processed_data_t2m{temp_var}.csv")
    df = df[df["Region"] == region]
    fig = px.line(df, x="Year", y="Average_Temperature", title=f"{region} {temp_var.capitalize()} Temperature", color="Dataset", color_discrete_map={"era5": "#33b1ff", "merra2": "#24a148"}, facet_col="Month", facet_col_wrap=4)
    fig.update_layout(legend_title_text="Dataset")
    fig.for_each_trace(lambda t: t.update(name=t.name.upper()))
    fig.for_each_yaxis(lambda yaxis: yaxis.update(title_text=None))
    fig.update_yaxes(title_text=f"Local Average Temperature (°C)", row = 2, col = 1)

    return fig

if __name__ == "__main__":
    if plot_type == "global":
        fig = global_average_temperature(temp_var)
    elif plot_type == "local":
        fig = local_average_temperature(temp_var, region)
        
    fig.show()