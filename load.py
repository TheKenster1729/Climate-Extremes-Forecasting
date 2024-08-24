import pandas as  pd
import numpy as np
from netCDF4 import Dataset
import os
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors

def get_data(file_path):
    dataset = Dataset(file_path)
    return dataset.RangeBeginningDate, dataset["lon"][:], dataset["lat"][:], dataset["T2MMAX"][0, :, :]

def make_dataframe(file_path):
    time, lon, lat, temp = get_data(file_path)
    # Create a DataFrame
    df = pd.DataFrame(columns = lon, index = lat, data = temp)
    return time, df

fig = go.Figure()
colors = n_colors((255, 0, 0), (0, 255, 0), 2022-1980+1)
for i, year in enumerate(range(1980, 2023)):
    year_df = pd.DataFrame(columns = ["Date", "Temperature"])
    path = r"MERRA2/Temperature Data/Max Temp/" + str(year)
    for file in os.listdir(path):
        if file[-2:] == "nc":
            file_path = os.path.join(path, file)
            date, df = make_dataframe(file_path)
            data_year, data_month, data_day = date.split("-")
            data_date = datetime.datetime(int(data_year), int(data_month), int(data_day))

            daily_average_temp_boston = df.iloc[265, 174]-273.15
            df_to_add = pd.DataFrame({
                "Date": [data_date],
                "Temperature": [daily_average_temp_boston]
            })
            year_df = pd.concat([year_df, df_to_add], ignore_index=True, axis = 0)

    year_df.sort_values(by = "Date", inplace = True)
    fig.add_trace(go.Scatter(x=year_df["Date"].dt.dayofyear, y=year_df["Temperature"], mode="lines", name=str(year), line=dict(width=0.5, color="rgb"+str(colors[i]))))

fig.show()