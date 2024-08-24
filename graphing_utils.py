import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
import pandas as pd
import json
from collections import OrderedDict
import numpy as np
from pprint import pprint
from data_retrieval import DataRetrieval, CollectRegionalData

class DailyHeatMap:
    def __init__(self, year, day_of_year, data_directory = r"MERRA2/Temperature Data/Clean Max Temp"):
        self.year = year
        self.data_directory = data_directory

    def make_dataframe(self):
        files = os.listdir(self.data_directory)
        df = pd.DataFrame()
        for file in files:
            if re.match("^[0-9]", file):
                lat_and_lon = file[:-4]
                lat, lon = lat_and_lon.split("_")
                lat, lon = float(lat), float(lon)

                df_to_append = pd.read_csv(os.path.join(self.data_directory, file))
                df_to_append["Year"] = self.year
                df_to_append["Lat"] = lat
                df_to_append["Lon"] = lon

                df = pd.concat([df, df_to_append])

        return df

class TimeSeriesForOneRegion(DataRetrieval):
    def __init__(self, metric = "average", region = "global", var = "T2MMAX", polygon_data = None):
        super().__init__(metric = metric, region = region, var = var, polygon_data = polygon_data)

    def make_full_timeseries_plot_together(self):
        """
        Makes a timeseries plot of the temperature data for a given region. Shows the entire timeseries, not segregated by year.
        """
        data = self.get_data_by_month()
        temps = []
        dates = []
        for year in data.keys():
            for month in data[year].keys():
                temps += data[year][month]
                dates += [f"{year}-{month}-{i + 1}" for i in range(len(data[year][month]))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=temps, mode="lines",
                                 text=dates,  # Hover text with the actual date
                                 hovertemplate='Date: %{text}<br>%{y:.2f}°C'
                                 ))
        fig.show()

    def make_full_timeseries_plot_by_year(self):
        """
        Makes a timeseries plot of the temperature data for a given region, segregated by year.
        """
        data = self.get_data_by_month()
        fig = go.Figure()
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for year in [str(i) for i in self.years]:
            temps_for_year = []
            dates_for_year = []
            for month in months:
                # doing it this way ensures that the months come out in the correct order on the graph
                # because they may not be ordered properly in the json file
                temps_for_year += data[year][month]
                dates_for_year += [f"{month}-{i + 1}" for i in range(len(data[year][month]))]
            color = "rgb(105, 105, 105)" if str(year) != "2022" else "rgb(255, 165, 0)"
            width = 0.5 if str(year) != "2022" else 4
            fig.add_trace(go.Scatter(x=dates_for_year, y=temps_for_year, mode="lines",
                                     text=dates_for_year,  # Hover text with the actual date
                                     hovertemplate='Date: %{text}<br>%{y:.2f}°C',
                                     name = year,
                                     line = dict(color = color, width = width)
                                     ))
        fig.show()

    def generate_statistics_df(self):
        if self.file is None:
            file = os.path.join(self.data_directory, f"data_{self.lat}_{self.lon}.json")
        else:
            file = self.file
        data = json.load(open(file, "r"))
        sorted_data = {i: data[str(i)] for i in sorted([int(i) for i in list(data.keys())])} # ensuring years are ordered correctly
        statistics_data = {}
        for year in sorted_data.keys():
            year_data = sorted_data[year]
            statistics_data[year] = {}
            for month in year_data.keys():
                month_data = year_data[month]
                month_avg = sum(month_data) / len(month_data)
                month_median = np.median(month_data)
                month_std = np.std(month_data)
                month_max = max(month_data)
                month_min = min(month_data)
                statistics_data[year][month] = {"avg": month_avg, "median": month_median, "std": month_std, "max": month_max, "min": month_min}

        statistics_df = pd.DataFrame()
        statistics_df["Year"] = [i for i in statistics_data.keys() for j in range(12)]
        statistics_df["Month"] = [j for i in statistics_data.keys() for j in statistics_data[i].keys()]
        statistics_df["Avg"] = [statistics_data[i][j]["avg"] for i in statistics_data.keys() for j in statistics_data[i].keys()]
        statistics_df["Median"] = [statistics_data[i][j]["median"] for i in statistics_data.keys() for j in statistics_data[i].keys()]
        statistics_df["Std"] = [statistics_data[i][j]["std"] for i in statistics_data.keys() for j in statistics_data[i].keys()]
        statistics_df["Max"] = [statistics_data[i][j]["max"] for i in statistics_data.keys() for j in statistics_data[i].keys()]
        statistics_df["Min"] = [statistics_data[i][j]["min"] for i in statistics_data.keys() for j in statistics_data[i].keys()]

        return statistics_df
    
    def make_statistics_plot(self, title):
        statistics_df = self.generate_statistics_df()

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["Avg", "Median", "Max", "Min"])
        colors = ["rgb(31, 119, 180)", "rgb(255, 127, 14)", "rgb(44, 160, 44)", "rgb(214, 39, 40)", "rgb(148, 103, 189)", "rgb(140, 83, 102)", "rgb(23, 190, 207)", "rgb(188, 128, 189)", "rgb(23, 190, 207)", "rgb(140, 83, 102)", "rgb(23, 190, 207)", "rgb(140, 83, 102)"]
        for i, month in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):
            fig.add_trace(go.Scatter(x=statistics_df[statistics_df["Month"] == month]["Year"], y=statistics_df[statistics_df["Month"] == month]["Avg"], mode="lines", name=month, legendgroup=month, marker=dict(color=colors[i])), row=1, col=1)
            fig.add_trace(go.Scatter(x=statistics_df[statistics_df["Month"] == month]["Year"], y=statistics_df[statistics_df["Month"] == month]["Median"], mode="lines", name=month, legendgroup=month, marker=dict(color=colors[i])), row=1, col=2)
            fig.add_trace(go.Scatter(x=statistics_df[statistics_df["Month"] == month]["Year"], y=statistics_df[statistics_df["Month"] == month]["Max"], mode="lines", name=month, legendgroup=month, marker=dict(color=colors[i])), row=2, col=1)
            fig.add_trace(go.Scatter(x=statistics_df[statistics_df["Month"] == month]["Year"], y=statistics_df[statistics_df["Month"] == month]["Min"], mode="lines", name=month, legendgroup=month, marker=dict(color=colors[i])), row=2, col=2)
        
        fig.update_layout(title=title,
                          xaxis_title="Year",
                          yaxis_title="Temperature (°C)")
        fig.show()

class TempDistributionsForOneRegion:
    def __init__(self, lat, lon, data_directory = r"MERRA2/Temperature Data/"):
        self.lat = lat
        self.lon = lon
        self.data_directory = data_directory

    def make_distribution_for_one_year(self, year: str):
        file = os.path.join(self.data_directory, f"data_{self.lat}_{self.lon}.json")
        data = json.load(open(file, "r"))
        sorted_data = {i: data[str(i)] for i in sorted([int(i) for i in list(data.keys())])} # ensuring years are ordered correctly
        data_for_this_year = sorted_data[int(year)]

        fig = make_subplots(rows = 3, cols = 4, subplot_titles = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        for i, month in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):
            fig.add_trace(go.Histogram(x = data_for_this_year[month]), row = i // 4 + 1, col = i % 4 + 1)
        fig.show()

class TimeSeriesFromAppData(CollectRegionalData):
    def __init__(self, polygon_data, plot_title, future = "acc"):
        super().__init__(polygon_data = polygon_data)
        self.plot_title = plot_title
        self.future = future

    def make_timeseries_plot(self):
        data = self.aggregate_inside_points_temp_data()

        fig = go.Figure()
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for year in [str(i) for i in self.years]:
            temps_for_year = []
            dates_for_year = []
            for month in months:
                # doing it this way ensures that the months come out in the correct order on the graph
                # because they may not be ordered properly in the json file
                temps_for_year += data[year][month]
                dates_for_year += [f"{month}-{i + 1}" for i in range(len(data[year][month]))]
            color = "rgb(105, 105, 105)" if str(year) != "2022" else "rgb(255, 165, 0)"
            width = 0.5 if str(year) != "2022" else 4
            fig.add_trace(go.Scatter(x=dates_for_year, y=temps_for_year, mode="lines",
                                     text=dates_for_year,  # Hover text with the actual date
                                     hovertemplate='Date: %{text}<br>%{y:.2f}°C',
                                     name = year,
                                     line = dict(color = color, width = width)
                                     ))
        fig.update_layout(height = 800, width = 1000, title = self.plot_title, xaxis_title = "Date", yaxis_title = "Temperature (°C)")
        return fig

class TimeSeriesFromExistingRegion:
    def __init__(self, region, var, aggregation = None):
        # expects region to be a tuple of (lat, lon) or a string
        self.region_dict = {"New England": "areaweightednewengland", "Global": "global"}
        self.raw_region = region
        self.region = self.region_dict[region] if region in self.region_dict else region
        self.var = var
        self.aggregation = aggregation

        self.path_to_region_data = r"MERRA2/JSON Files/Coordinates" if type(self.region) == tuple else r"MERRA2/JSON Files/Regional Aggregates"
        
        if type(self.region) == tuple:
            self.path_to_file = os.path.join(self.path_to_region_data, f"data_{self.region[0]}_{self.region[1]}_{self.var}.json")
        else:
            self.path_to_file = os.path.join(self.path_to_region_data, f"{self.region}_{self.aggregation}_{self.var}.json")

    def load_region_data(self):
        return json.load(open(self.path_to_file, "r"))

if __name__ == "__main__":
    # data_directory = r"MERRA2/Temperature Data/"
    # data = TimeSeriesForOneRegion(42.5, -71.875, data_directory = data_directory, file = r"Regional Averages/global_averages.json").make_statistics_plot("Aggregate Statistics for World")
    # TimeSeriesForOneRegion(region = "newengland").make_full_timeseries_plot_by_year()
    # polygon_data = json.load(open(r"Regional Geojsons/lowermississippiregion.geojson", "r"))["features"][0]["geometry"]
    # fig = TimeSeriesFromAppData(polygon_data, "Lower Mississippi Region").make_timeseries_plot()
    # fig.show()
    # time_series = TimeSeriesForOneRegion(region = "newengland").make_full_timeseries_plot_together()
    TimeSeriesForOneRegion().make_full_timeseries_plot_by_year()
