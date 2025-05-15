<<<<<<< HEAD
=======
<<<<<<< HEAD
import statsmodels.formula.api as smf
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import json
import pandas as pd
import numpy as np

def complete_df(region):
    era5_data = json.load(open(r"ERA5/Temperature Data/JSON Files/us-states-era5-t2m-rescaled.json"))
    merra2_data = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/us-states-regions.json"))
    # regions_in_common = list(set(era5_data["contains"]) & set(merra2_data["data"]))

    all_data = []
    month_identifier = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    era5_results = era5_data["data"][region]["results"]
    merra2_results = merra2_data["data"][region]["results"]
    for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
        for year in range(1980, 2023):
            year_month_average_era5 = sum(era5_results[str(year)][month])/len(era5_results[str(year)][month])
            year_month_average_merra2 = sum(merra2_results[str(year)][month])/len(merra2_results[str(year)][month])
            all_data.append([year, month, year_month_average_era5, "era5"])
            all_data.append([year, month, year_month_average_merra2, "merra2"])

    df = pd.DataFrame(all_data, columns = ["Year", "Month", "Average_Temperature", "Dataset"])

    era5_global_temps = json.load(open(r"ERA5/Temperature Data/JSON Files/world-average.json"))
    era5_global_temps_dict = {}
    for entry in era5_global_temps:
        year = entry["name"]
        if 1980 <= int(year) <= 2022:
            temps_list = [i for i in entry["data"] if i]
            era5_global_temps_dict[int(year)] = sum(temps_list)/len(temps_list)
    
    merra2_global_temps = pd.read_csv(r"global_average_temp_by_year.csv")
    merra2_global_temps_dict = {}
    for index, row in merra2_global_temps.iterrows():
        year = row["Year"]
        if 1980 <= int(year) <= 2022:
            merra2_global_temps_dict[int(year)] = row["Average"]
    
    full_dict = {"era5": era5_global_temps_dict, "merra2": merra2_global_temps_dict}

    df["Global_Temp"] = df.apply(lambda row: full_dict[row["Dataset"]][row["Year"]], axis = 1)

    return df

region_data = []
df = complete_df("MA")
# for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
#     era5_df = df[(df["Dataset"] == "era5") & (df["Month"] == month)].reset_index(drop = True)
#     merra2_df = df[(df["Dataset"] == "merra2") & (df["Month"] == month)].reset_index(drop = True)

#     n_boot = 2
#     slopes_ERA5 = []
#     slopes_MERRA2 = []
#     intercepts_ERA5 = []
#     intercepts_MERRA2 = []
#     for i in range(n_boot):
#         # Create bootstrap samples that preserve (year, month) pairing.
#         sample_idx = np.random.choice(era5_df.index, size=len(era5_df), replace=True)
#         sample_era5 = era5_df.loc[sample_idx]
#         sample_merra2 = merra2_df.loc[sample_idx]
#         print(sample_era5)
#         print(sample_merra2)
#         print("---")
=======
>>>>>>> 2e2e3d2b (WIP)
import dash
from flask import Flask
import dash_bootstrap_components as dbc
from dash import html, dcc, _dash_renderer
from analysis import RiskAssessment
import dash_mantine_components as dmc

_dash_renderer._set_react_version("18.2.0")

server = Flask(__name__)
app = dash.Dash(__name__, server = server, external_stylesheets = [dbc.themes.BOOTSTRAP])

app.layout = dmc.MantineProvider([RiskAssessment(dataset = "ERA5", var = "T2MMAX", state = "MA").risk_assessment_div_element()])

if __name__ == "__main__":
<<<<<<< HEAD
    app.run_server(debug = True)
=======
    app.run_server(debug = True)
>>>>>>> be14593b33a08ae29df60822b7dcc702a6d70f62
>>>>>>> 2e2e3d2b (WIP)
