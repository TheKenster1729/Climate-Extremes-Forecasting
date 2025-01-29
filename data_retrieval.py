import paramiko
from scp import SCPClient
import os
from netCDF4 import Dataset
import netCDF4 as nc
import pandas as  pd
import datetime
import numpy as np
from pprint import pprint
import json
from shapely.geometry import Point, shape
import xarray as xr
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib.pyplot as plt
import time

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def download_data(password):
    ssh = createSSHClient("svante.mit.edu", 22, "kcox1729", password)

    # SCPCLient takes a paramiko transport as an argument
    scp = SCPClient(ssh.get_transport())

    for year in range(1980, 2021):
        scp.get('/net/fs01/data/MERRA2/daily/M2SDNXSLV.5.12.4/' + str(year), recursive=True, local_path="./MERRA2/Temperature Data/Max Temp")

    scp.close()

def get_data(file_path):
    dataset = Dataset(file_path)
    return dataset.RangeBeginningDate, dataset["lon"][:], dataset["lat"][:], dataset["T2MMAX"][0, :, :]

def make_dataframe(file_path):
    time, lon, lat, temp = get_data(file_path)
    # Create a DataFrame
    df = pd.DataFrame(columns = lon, index = lat, data = temp)
    return time, df

def move_relevant_data_to_csv(output_folder, data_path, geo_location, lat_index, long_index):
    df_to_convert = pd.DataFrame()
    df_to_convert["Day of Year"] = [i for i in range(1, 366)]
    for year in range(1980, 2023):
        year_df = pd.DataFrame(columns = ["Date", "Temperature"])
        path = r"MERRA2/Temperature Data/Max Temp/" + str(year)
        for file in os.listdir(path):
            if file[-2:] == "nc":
                file_path = os.path.join(path, file)
                date, df = make_dataframe(file_path)
                data_year, data_month, data_day = date.split("-")
                data_date = datetime.datetime(int(data_year), int(data_month), int(data_day))

                daily_average_temp_boston = df.iloc[lat_index, long_index]-273.15
                df_to_add = pd.DataFrame({
                    "Date": [data_date],
                    "Temperature": [daily_average_temp_boston]
                })
                year_df = pd.concat([year_df, df_to_add], ignore_index=True, axis = 0)

        # Convert Date column to string for regex matching
        year_df["Date_str"] = year_df["Date"].astype(str)
        year_df = year_df.sort_values(by="Date").drop(index=year_df[year_df["Date_str"].str.contains('^\d{4}-02-29$', regex=True)].index)
        year_df = year_df.drop(columns=["Date_str"])  # Drop the temporary string column
        df_to_convert[year] = year_df["Temperature"].values
        df_to_convert.sort_values(by = "Day of Year", inplace = True)

    df_to_convert.to_csv("./MERRA2/Temperature Data/Clean Max Temp/" + geo_location + ".csv")

# converting to one big netcdf file
def do_the_big_convert():
    # Paths
    base_path = r'MERRA2/Temperature Data/Max Temp'  # Path to the folder containing yearly folders

    # Initialize lists to store data
    temperature_data = []

    # Loop through each year
    for year in range(1984, 1988):
        year_path = os.path.join(base_path, str(year))
        
        # Loop through each day in the year
        for day_file in sorted(os.listdir(year_path)):
            if day_file.endswith('.nc'):
                day_path = os.path.join(year_path, day_file)
                
                # Open the NetCDF file
                with nc.Dataset(day_path, 'r') as ds:
                    # Assuming variable name is 'temperature', adjust if different
                    daily_temperature = ds.variables['T2MMAX'][:]
                    temperature_data.append(daily_temperature[0])

    # Convert the list to a numpy array
    temperature_data = np.array(temperature_data)

    # Define the dimensions
    num_days = temperature_data.shape[0]
    num_lat = temperature_data.shape[1]
    num_lon = temperature_data.shape[2]

    # Create a new NetCDF file
    output_path = r'MERRA2/Temperature Data/Clean Max Temp/combined_data_test.nc'
    with nc.Dataset(output_path, 'w', format='NETCDF4') as dst:
        # Create dimensions
        dst.createDimension('time', num_days)
        dst.createDimension('lat', num_lat)
        dst.createDimension('lon', num_lon)
        
        # Create variables
        times = dst.createVariable('time', 'i4', ('time',))
        lats = dst.createVariable('lat', 'f4', ('lat',))
        lon = dst.createVariable('lon', 'f4', ('lon',))
        temps = dst.createVariable('maxdailytemperature2m', 'f4', ('time', 'lat', 'lon'))
        
        # Assign data to variables
        times[:] = np.arange(num_days)
        lats[:] = np.linspace(start = -90, stop = 90, num = 361)
        lon[:] = np.linspace(start = -180, stop = 179.375, num = 576)
        temps[:] = temperature_data

    print(f'Combined data saved to {output_path}')

def lat_lon_to_json(lat, lon):
    files = sorted(os.listdir(r"MERRA2/Temperature Data/Max Temp"))
    months = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}
    days_in_month_regular = {"Jan": 31, "Feb": 28, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30, "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31}
    days_in_month_leap = {"Jan": 31, "Feb": 29, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30, "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31}
    
    data = {}
    for folder in files[1:]:
        data[int(folder)] = {}
        for file in os.listdir(r"MERRA2/Temperature Data/Max Temp/{}".format(folder)):
            time, df = make_dataframe("MERRA2/Temperature Data/Max Temp/{}/{}".format(folder, file))
            year, month, day = time.split("-")
            temp_data = df.loc[lat, lon] - 273.15
            month_name = months[month]
            int_day = int(day)

            # ensuring days of the month are ordered correctly - using os.listdir does not guarantee order
            if ((int(year) % 4 == 0 and int(year) % 100 != 0) or (int(year) % 400 == 0)):
                days_in_month = days_in_month_leap[month_name]
            else:
                days_in_month = days_in_month_regular[month_name]
            if data[int(folder)].get(month_name):
                data[int(folder)][month_name][int_day - 1] = temp_data
            else:
                data[int(folder)][month_name] = [0 for i in range(days_in_month)]
                data[int(folder)][month_name][int_day - 1] = temp_data

    json.dump(data, open(r"MERRA2/Temperature Data/data_{}_{}.json".format(lat, lon), "w"))

class RetrieveSingleVariable:
    def __init__(self, password, data_location, output_path, var):
        self.password = password
        self.data_location = data_location
        self.output_path = output_path
        self.var = var

    def createSSHClient(self, server, port, user):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, port, user, self.password)

        return client

    def download_data(self):
        ssh = self.createSSHClient("svante.mit.edu", 22, "kcox1729")

        # SCPCLient takes a paramiko transport as an argument
        scp = SCPClient(ssh.get_transport())
        path_to_data = os.path.join("/net/fs01/data/MERRA2", self.data_location)

        for year in range(1980, 2023):
            scp.get(os.path.join(path_to_data, str(year)), recursive=True, local_path=self.output_path)

            for input_file in os.listdir(os.path.join(self.output_path, str(year))):
                if input_file.endswith(".nc"):
                    with Dataset(os.path.join(self.output_path, str(year), input_file), 'r') as src:
                        var = src.variables[self.var]
                        var_dims = var.dimensions
                        coord_vars = [dim for dim in var_dims if dim in src.variables]
                        date = src.RangeBeginningDate

                        with Dataset(os.path.join(self.output_path, str(year), date + ".nc"), 'w') as dst:
                            # Copy dimensions
                            for dim_name in var_dims:
                                dim_length = len(src.dimensions[dim_name])
                                dst.createDimension(dim_name, dim_length)

                            # Copy coordinate variables
                            for coord in coord_vars:
                                src_var = src.variables[coord]
                                dst_var = dst.createVariable(coord, src_var.datatype, src_var.dimensions)
                                dst_var[:] = src_var[:]
                                for attr in src_var.ncattrs():
                                    dst_var.setncattr(attr, src_var.getncattr(attr))

                            # Copy the variable of interest
                            dst_var = dst.createVariable(self.var, var.datatype, var.dimensions)
                            dst_var[:] = var[:]

                            # Optionally, copy global attributes
                            for attr in src.ncattrs():
                                dst.setncattr(attr, src.getncattr(attr))

                os.remove(os.path.join(self.output_path, str(year), input_file))

class DataRetrieval:
    def __init__(self, metric = "average", region = "global", var = "T2MMAX", polygon_data = None):
        self.metric = metric
        self.region = region
        self.var = var
        self.polygon_data = polygon_data
        self.years = [i for i in range(1980, 2023)]
        self.lats = np.linspace(start = -90, stop = 90, num = 361)
        self.lons = np.linspace(start = -180, stop = 179.375, num = 576)
        self.weights = np.cos(np.radians(self.lats)) # in case an area-weighted average needs to be calculated
        self.data_directory = r"MERRA2/Temperature Data/Max Temp"

        if self.metric not in ["average", "median", "maximum", "minimum", "std", "values"]:
            raise ValueError("Invalid metric. Please choose from 'average', 'median', 'maximum', 'minimum', 'values', or 'std'.")

        if type(self.region) is tuple:
            self.lat = self.region[0]
            self.lon = self.region[1]
        else:
            self.lat = None
            self.lon = None

    def json_exists(self):
        if self.lat:
            # the region is latitude, longitude coordinate (not an aggregate)
            possible_path = os.path.join(r'MERRA2/JSON Files/Coordinates', "data_{}_{}_{}.json".format(self.lat, self.lon, self.var.lower()))
            if os.path.exists(possible_path):
                return possible_path
            else:
                return None
        else:
            # the region is an aggregate
            possible_path = os.path.join(r'MERRA2/JSON Files/Regional Aggregates', "{}_{}_{}.json".format(self.region, self.metric, self.var.lower()))
            if os.path.exists(possible_path):
                return possible_path
            else:
                return None

    def create_json(self):
        if self.region == "global":
            temperature_dict = {}

            for year in self.years:
                temperature_dict[year] = {month: [] for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}

                for file in os.listdir(os.path.join(self.data_directory, str(year))):
                    if file.endswith(".nc"):
                        data = xr.open_dataset(os.path.join(self.data_directory, str(year), file))
                        data_values = data[self.var].values
                        date_str = data.RangeBeginningDate

                        if self.metric == "average":
                            # calculate the area weighted average
                            weights = np.average(data_values, weights = self.weights, axis = 1)
                            to_add = weights.mean() - 273.15
                        else:
                            # other metrics
                            pass

                        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                        month_name = date.strftime("%b")
                        day_of_month = date.day

                        # Ensure the list for the month is large enough
                        while len(temperature_dict[year][month_name]) < day_of_month:
                            temperature_dict[year][month_name].append(None)
                        
                        # Store the average temperature in the appropriate position
                        temperature_dict[year][month_name][day_of_month - 1] = to_add

                print(f"Finished year {year}")

            json.dump(temperature_dict, open(os.path.join(r"MERRA2/JSON Files/Regional Aggregates", "global_{}_{}.json".format(self.metric, self.var.lower())), "w"))

        elif type(self.region) is tuple:
            # no aggregation necessary
            pass
        else:
            # assume region is a polygon
            # I already have code for this, so reusing it here
            CollectRegionalData(polygon_file = self.region, data_directory = self.data_directory, polygon_directory = self.polygon_directory).aggregate_inside_points_temp_data()
            pass

    def get_data_by_month(self):
        # first look if a json file has already been created with this data, create it if not for faster loading next time
        path_to_load = self.json_exists()
        if path_to_load:
            self.data = json.load(open(path_to_load, "r"))
        else:
            print("File not located. Creating new json file.")
            self.create_json()
            path_to_load = self.json_exists()
            self.data = json.load(open(path_to_load, "r"))

        return self.data

class GenerateDataFrame:
    def __init__(self, lat = None, lon = None, json = None, aggregation: {"avg", "median", "maximum", "minimum", "std", None} = None):
        if json:
            self.file = json
        else:
            self.file = r"MERRA2/Temperature Data/data_{}_{}.json".format(lat, lon)

        if not self.file:
            raise ValueError("Either a lat, lon, or json must be provided")
        
        self.aggregation = aggregation
    
    def return_data(self):
        if self.aggregation:
            # aggregation is done on a monthly basis
            pass
        else:
            # assume user is only asking for raw data
            pass

class CollectRegionalDataFromFile:
    def __init__(self, polygon_file, data_directory = r"MERRA2/Temperature Data/Max Temp", polygon_directory = r"Regional Geojsons"):
        self.data_directory = data_directory
        self.polygon_directory = polygon_directory
        self.polygon_file = polygon_file
        self.polygon_path = os.path.join(polygon_directory, polygon_file)
        self.lat_points = np.linspace(start = -90, stop = 90, num = 361)
        self.lon_points = np.linspace(start = -180, stop = 179.375, num = 576)
        self.years = [i for i in range(1980, 2023)]

        # Load the GeoJSON file with the drawn polygon
        with open(self.polygon_path) as f:
            geojson_data = json.load(f)

        # Extract coordinates
        self.polygon = shape(geojson_data['features'][0]['geometry'])

    def get_data(self, file_path):
        dataset = Dataset(file_path)

        return dataset.RangeBeginningDate, dataset["lon"][:], dataset["lat"][:], dataset["T2MMAX"][0, :, :]

    def make_dataframe(self, file_path):
        time, lon, lat, temp = self.get_data(file_path)
        # Create a DataFrame
        df = pd.DataFrame(columns = lon, index = lat, data = temp)

        return time, df

    def point_in_polygon(self, lon, lat):
        point = Point(lon, lat)

        return self.polygon.contains(point)

    def identify_inside_points(self):
        inside_points = []
        for lat in self.lat_points:
            for lon in self.lon_points:
                if self.point_in_polygon(lon, lat):
                    inside_points.append((lat, lon))

        print("Number of points inside polygon: ", len(inside_points))
        return inside_points

    def aggregate_inside_points_temp_data(self):
        inside_points = self.identify_inside_points()
        results_dict = {}
        months = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}
        days_in_month_regular = {"Jan": 31, "Feb": 28, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30, "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31}
        days_in_month_leap = {"Jan": 31, "Feb": 29, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30, "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31}

        # iterate through each year in the data set
        for year in self.years:
            results_dict[str(year)] = {}
            # each file corresponds to one day
            for file in os.listdir(os.path.join(self.data_directory, str(year))):
                time, df = self.make_dataframe(os.path.join(self.data_directory, str(year), file))
                _, month, day = time.split("-")
                this_day_in_region = []
                # iterate through each point in the region and store it in the list
                for point in inside_points:
                    lat, lon = point
                    this_day_in_region.append(df.loc[lat, lon] - 273.15)

                month_name = months[month]
                # ensuring days of the month are ordered correctly - using os.listdir does not guarantee order
                if ((int(year) % 4 == 0 and int(year) % 100 != 0) or (int(year) % 400 == 0)):
                    days_in_month = days_in_month_leap[month_name]
                else:
                    days_in_month = days_in_month_regular[month_name]

                # now take the average for this day and store it in the results dict
                if results_dict[year].get(month_name):
                    results_dict[year][month_name][int(day) - 1] = sum(this_day_in_region)/len(this_day_in_region)
                else:
                    results_dict[year][month_name] = [0 for i in range(days_in_month)]
                    results_dict[year][month_name][int(day) - 1] = sum(this_day_in_region)/len(this_day_in_region)

            print(f"Finished year {year}")

        json.dump(results_dict, open(r"Regional Averages/{}".format(self.polygon_file[:-8] + ".json"), "w"))

class CollectRegionalData:
    def __init__(self, polygon_data, data_directory = r"MERRA2/Temperature Data/Max Temp", var = "T2MMAX", from_app = False, name = None, write_json = True, write_location = r"MERRA2/JSON Files/Regional Aggregates/"):
        self.from_app = from_app
        self.write_json = write_json
        self.write_location = write_location
        self.var = var
        
        if self.from_app:
            self.polygon_data = polygon_data
            self.polygon_file_name = name
        else:
            assert type(polygon_data) == str
            try:
                self.polygon_data = json.load(open(polygon_data, "r"))["features"][0]["geometry"]
            except KeyError:
                self.polygon_data = json.load(open(polygon_data, "r"))

            self.polygon_file_name = os.path.basename(polygon_data)

        self.data_directory = data_directory

        # Extract coordinates
        self.polygon = shape(self.polygon_data)

    def get_data(self, file_path):
        dataset = Dataset(file_path)
        values = dataset["time"][:][0], dataset["lon"][:], dataset["lat"][:], dataset[self.var][0, :, :]
        dataset.close()

        return values

    def make_dataframe(self, file_path):
        time, lon, lat, temp = self.get_data(file_path)
        # Create a DataFrame
        df = pd.DataFrame(columns = lon, index = lat, data = temp)

        return time, df

    def point_in_polygon(self, lon, lat):
        point = Point(lon, lat)

        return self.polygon.contains(point)

    def identify_inside_points(self, lat_points, lon_points):
        inside_points = []
        for lat in lat_points:
            for lon in lon_points:
                if self.point_in_polygon(lon, lat):
                    inside_points.append((lat, lon))

        print("Number of points inside polygon: ", len(inside_points))
        return inside_points

    def aggregate_inside_points_temp_data(self, lat_points = None, lon_points = None):
        inside_points = self.identify_inside_points(lat_points, lon_points)
        results_dict = {}
        months = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}
        days_in_month_regular = {"Jan": 31, "Feb": 28, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30, "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31}
        days_in_month_leap = {"Jan": 31, "Feb": 29, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30, "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31}

        if len(inside_points) == 0:
            return None, None
        else:
            # just need to create weights once
            weights = []
            for lat, lon in inside_points:
                weights.append(np.cos(np.radians(lat)))
            weights /= np.array(weights).mean()

            # iterate through each year in the data set
            for year in sorted(os.listdir(self.data_directory)):
                results_dict[year] = {}
                # each file corresponds to one day
                directory = os.path.join(self.data_directory, year)
                if not os.path.isdir(directory):
                    continue
                for file in os.listdir(directory):
                    time, data_lon, data_lat, temp = self.get_data(os.path.join(self.data_directory, year, file))
                    _, month, day = file[:-3].split("-")
                    # iterate through each point in the region and store it in the list
                    this_day_in_region = []
                    for lat, lon in inside_points:
                        # very unfortunate bug at the poles - the actual lat/lon point is like 1e-13
                        # instad of 0
                        # so we find the temperatures by the index
                        lat_index = np.argwhere(data_lat == lat)[0][0]
                        lon_index = np.argwhere(data_lon == lon)[0][0]
                        this_day_in_region.append(temp[lat_index, lon_index] - 273.15)
                    this_day_in_region = np.array(this_day_in_region)
                    this_day_in_region_avg = np.average(this_day_in_region, weights = weights)

                    month_name = months[month]
                    # ensuring days of the month are ordered correctly - using os.listdir does not guarantee order
                    if ((int(year) % 4 == 0 and int(year) % 100 != 0) or (int(year) % 400 == 0)):
                        days_in_month = days_in_month_leap[month_name]
                    else:
                        days_in_month = days_in_month_regular[month_name]

                    # now take the average for this day and store it in the results dict
                    if results_dict[year].get(month_name):
                        results_dict[year][month_name][int(day) - 1] = this_day_in_region_avg
                    else:
                        results_dict[year][month_name] = [0 for i in range(days_in_month)]
                        results_dict[year][month_name][int(day) - 1] = this_day_in_region_avg

                print(f"Finished year {year}")

            if self.write_json:
                json.dump(results_dict, open(self.write_location + "{}".format(self.polygon_file_name[:-8] + "_" + "average" + "_" + "t2mmax" + ".json"), "w"))

            return results_dict, len(inside_points)

class GlobalAverageByDay:
    def __init__(self, folder_of_files = r"MERRA2/Temperature Data/Max Temp") -> None:
        self.data_folder = folder_of_files
        self.years = [i for i in range(1980, 2023)]

    def get_data(self, file_path):
        dataset = Dataset(file_path)

        return dataset.RangeBeginningDate, dataset["lon"][:], dataset["lat"][:], dataset["t2m"][0, :, :]
    
    def make_dataframe(self, file_path):
        time, lon, lat, temp = self.get_data(file_path)
        # Create a DataFrame
        df = pd.DataFrame(columns = lon, index = lat, data = temp)

        return time, df
    
    def is_leap_year(self, year):
        return ((int(year) % 4 == 0 and int(year) % 100 != 0) or (int(year) % 400 == 0))
    
    def area_weighted_temperatures(self, df):
        """
        This function takes a dataframe of temperatures over latitude and longitude pairs and
        returns the area-weighted average temperature. The dataframe rows are longitudes
        and columns are latitudes.
        
        Parameters:
        df (pd.DataFrame): DataFrame where rows are longitudes and columns are latitudes.

        Returns:
        float: Area-weighted average temperature.
        """
        # Extract latitudes from the row index
        latitudes = np.radians(df.index.astype(float))

        # Calculate weights as cosine of latitudes
        weights = np.cos(latitudes)

        # Normalize weights
        weights /= np.array(weights).mean()

        # Apply weights to the temperature data
        weighted_temps = df.apply(lambda x: np.average(x, weights = weights), axis = 0)

        area_weighted_avg_temp = weighted_temps.mean()
        
        # # Calculate the area-weighted average temperature
        # area_weighted_avg_temp = weighted_temperatures.values.sum() / df.values.size

        return area_weighted_avg_temp

    def make_global_average_dict(self):
        data = {}
        days_in_month_regular = {"Jan": 31, "Feb": 28, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30, "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31}
        days_in_month_leap = {"Jan": 31, "Feb": 29, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30, "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31}
        num_to_month_name = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}
        for year in self.years:
            data[str(year)] = {}
            for file in os.listdir(os.path.join(self.data_folder, str(year))):
                time, df = self.make_dataframe(os.path.join(self.data_folder, str(year), file))
                _, month, day = time.split("-")
                if data[str(year)].get(num_to_month_name[month]):
                    data[str(year)][num_to_month_name[month]][int(day) - 1] = self.area_weighted_temperatures(df) - 273.15
                else:
                    days_in_month = days_in_month_leap[num_to_month_name[month]] if self.is_leap_year(year) else days_in_month_regular[num_to_month_name[month]]
                    data[str(year)][num_to_month_name[month]] = [0 for i in range(days_in_month)]
                    data[str(year)][num_to_month_name[month]][int(day) - 1] = self.area_weighted_temperatures(df) - 273.15

        return data

    def sort_and_return_global_average_dict_as_json(self):
        # including this because they way I usually do it is becoming annoying
        data = self.make_global_average_dict()
        sorted_data_dict = {}
        months = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}
        for year in data.keys():
            data[year] = sorted(data[year].items(), key=lambda x: x[0])
            sorted_data_dict[year] = {months[i]: j for i, j in data[year]}
            for month in sorted_data_dict[year]:
                sorted_data_dict[year][month] = sorted(sorted_data_dict[year][month].items(), key=lambda x: x[0])
                sorted_data_dict[year][month] = [i[1] for i in sorted_data_dict[year][month]]

            print(f"Finished year {year}")

        json.dump(sorted_data_dict, open(r"Regional Averages/global_averages.json", "w"))

class CountyLevelData:
    def __init__(self):
        self.base_url = r"https://www.ncei.noaa.gov/pub/data/daily-grids/v1-0-0/averages/"
        self.merra2_state_data = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/us-states-regions.json", "r"))
        self.states = self.merra2_state_data["contains"]
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def get_data(self, year, month):
        url = self.base_url + str(year) + "/"
        filename = r"tmax-{}{}-cty-scaled.csv".format(str(year), month)
        df = pd.read_csv(url + filename, header = None)

        return df
        
    def plot_merra2_state_data(self, state, year):
        fig = make_subplots(rows = 3, cols = 4, subplot_titles = self.months)
        state_data = self.merra2_state_data["data"][state]["results"][year]

        for i, month in enumerate(self.months):
            if i == 0:
                fig.add_trace(go.Scatter(x = [i for i in range(1, 32)], y = state_data[month], line = dict(color = "magenta"), legendgroup = "MERRA2", name = "MERRA2"), row = (i // 4) + 1, col = (i % 4) + 1)
            else:
                fig.add_trace(go.Scatter(x = [i for i in range(1, 32)], y = state_data[month], line = dict(color = "magenta"), legendgroup = "MERRA2", name = None), row = (i // 4) + 1, col = (i % 4) + 1)

        fig.update_layout(title = "Merra2 Temperatures for {} in {}".format(state, year), legend = dict(groupclick = "toggleitem"))
        
        return fig
    
    def get_aggregated_county_data(self, state, year, month):
        year_and_month_data = self.get_data(year, month)
        # just pull out the state
        year_and_month_data["state"] = year_and_month_data[2].apply(lambda x: x.split(":")[0])
        state_data_county_level = year_and_month_data[year_and_month_data["state"] == state]
        average_state_daily_temp = state_data_county_level[state_data_county_level.columns[6:-1]].mean(axis = 0).values

        # drop days that aren't included
        average_state_daily_temp = np.delete(average_state_daily_temp, np.where(average_state_daily_temp < -900))

        return average_state_daily_temp
    
    def make_comparison_plot(self, state, year):
        fig = self.plot_merra2_state_data(state, year)

        for i, month in enumerate(self.months):
            month_number = "0" + str(i + 1) if i + 1 < 10 else str(i + 1)
            average_state_daily_temp = self.get_aggregated_county_data(state, year, month_number)
            if i == 0:
                fig.add_trace(go.Scatter(x = [i for i in range(1, 32)], y = average_state_daily_temp, line = dict(color = "mediumblue"), legendgroup = "County Level Data", name = "County Level Data"), row = (i // 4) + 1, col = (i % 4) + 1)
            else:
                fig.add_trace(go.Scatter(x = [i for i in range(1, 32)], y = average_state_daily_temp, line = dict(color = "mediumblue"), legendgroup = "County Level Data", name = None), row = (i // 4) + 1, col = (i % 4) + 1)

        fig.update_layout(title = "County Level Data vs MERRA2 Data for {} in {}".format(state, year))

        return fig
    
    def concordance_by_month_year(self, state):
        pearson_correlations = []
        for year in range(1980, 1981):
            for i, month in enumerate(self.months):
                month_number = "0" + str(i + 1) if i + 1 < 10 else str(i + 1)
                county_data = self.get_aggregated_county_data(state, str(year), month_number)
                merra2_data = self.merra2_state_data["data"][state]["results"][str(year)][str(month)]

                if len(county_data) != len(merra2_data):
                    continue
                else:
                    pearson_correlation = np.corrcoef(county_data, merra2_data)[0, 1]
                    pearson_correlations.append(pearson_correlation)

        return pearson_correlations

    def all_concordance(self):
        min_correlations = []
        for state in self.states:
            pearson_correlations = self.concordance_by_month_year(state)

            min_corr = min(pearson_correlations)
            min_correlations.append(min_corr)

        fig = go.Figure()
        fig.add_trace(go.Bar(x = self.states, y = min_correlations, name = "Min Pearson Correlation"))
        fig.update_layout(title = "Min Pearson Correlation for Each State")

        return fig

class EPPARegionAggregation:
    def __init__(self):
        self.eppa_region_data = gpd.read_file("/Users/kcox1729/Documents/JP-Data-Viz/assets/Eppa countries/eppa6_regions_simplified.shp")
        self.results_dict = {"coverage": "eppa_regions", "contains": [], "variable": "T2MMAX", "data": {}}
        self.eppa_region_names = self.eppa_region_data["EPPA6_Regi"]

    def aggregate_inside_points_temp_data(self):
        for region in self.eppa_region_names:
            region_geometry = self.eppa_region_data[self.eppa_region_data["EPPA6_Regi"] == region]["geometry"].values[0]
            results, num_points = CollectRegionalData(polygon_data = region_geometry, write_json = False, from_app = True, name = "eppa_regions").aggregate_inside_points_temp_data()
            if num_points == 0:
                continue
            else:
                self.results_dict["contains"].append(region)
                self.results_dict["data"][region] = {}
                self.results_dict["data"][region]["results"] = results
                self.results_dict["data"][region]["num_points"] = num_points
                self.results_dict["data"][region]["centroid"] = [region_geometry.centroid.x, region_geometry.centroid.y]

        json.dump(self.results_dict, open(r"MERRA2/JSON Files/Regional Aggregates/eppa_regions.json", "w"))

class StateCMI:
    def __init__(self):
        self.area_data = pd.read_excel(r"/Users/kcox1729/Downloads/LND01.xls", sheet_name = "Sheet1")
        self.cmi_data = pd.read_csv(r"/Users/kcox1729/Downloads/Climate Moisture Index (1).csv")

        self.area_data['STCOU'] = self.area_data['STCOU'].astype(str)
        self.area_data['STCOU'] = self.area_data['STCOU'].str.zfill(5)

        self.cmi_data['id'] = self.cmi_data['id'].astype(str)
        self.cmi_data['id'] = self.cmi_data['id'].str.zfill(5)

        self.abbreviation_to_name = {
            # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#States.
            "AK": "Alaska",
            "AL": "Alabama",
            "AR": "Arkansas",
            "AZ": "Arizona",
            "CA": "California",
            "CO": "Colorado",
            "CT": "Connecticut",
            "DE": "Delaware",
            "FL": "Florida",
            "GA": "Georgia",
            "HI": "Hawaii",
            "IA": "Iowa",
            "ID": "Idaho",
            "IL": "Illinois",
            "IN": "Indiana",
            "KS": "Kansas",
            "KY": "Kentucky",
            "LA": "Louisiana",
            "MA": "Massachusetts",
            "MD": "Maryland",
            "ME": "Maine",
            "MI": "Michigan",
            "MN": "Minnesota",
            "MO": "Missouri",
            "MS": "Mississippi",
            "MT": "Montana",
            "NC": "North Carolina",
            "ND": "North Dakota",
            "NE": "Nebraska",
            "NH": "New Hampshire",
            "NJ": "New Jersey",
            "NM": "New Mexico",
            "NV": "Nevada",
            "NY": "New York",
            "OH": "Ohio",
            "OK": "Oklahoma",
            "OR": "Oregon",
            "PA": "Pennsylvania",
            "RI": "Rhode Island",
            "SC": "South Carolina",
            "SD": "South Dakota",
            "TN": "Tennessee",
            "TX": "Texas",
            "UT": "Utah",
            "VA": "Virginia",
            "VT": "Vermont",
            "WA": "Washington",
            "WI": "Wisconsin",
            "WV": "West Virginia",
            "WY": "Wyoming",
            # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#Federal_district.
            "DC": "D.C.",
            # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#Inhabited_territories.
            "AS": "American Samoa",
            "GU": "Guam GU",
            "MP": "Northern Mariana Islands",
            "PR": "Puerto Rico PR",
            "VI": "U.S. Virgin Islands",
        }
        self.name_to_abbreviation = {v: k for k, v in self.abbreviation_to_name.items()}

    def get_data(self):
        area_and_cmi_df = self.cmi_data.join(self.area_data.set_index('STCOU'), on = 'id', how = 'inner')

        data = []
        for state in area_and_cmi_df["state"].unique():
            state_area = area_and_cmi_df[area_and_cmi_df["state"] == state]["LND010200D"].sum()

            area_weighted_cmi = area_and_cmi_df[area_and_cmi_df["state"] == state]["value"] * area_and_cmi_df[area_and_cmi_df["state"] == state]["LND010200D"]/state_area

            state_abbreviation = self.name_to_abbreviation[state]
            data.append([state, state_abbreviation, round(area_weighted_cmi.sum(), 4)])

        data_df = pd.DataFrame(data, columns = ["state", "abbreviation", "cmi"])

        data_df.to_csv(r"state_cmi.csv", index = False)

class ERA5Data:
    def __init__(self):
        self.data_folder = r"ERA5/Temperature Data/"

    def download_data(self, year):
        import cdsapi

        dataset = "derived-era5-single-levels-daily-statistics"
        request = {
            "product_type": "reanalysis",
            "variable": ["2m_temperature"],
            "year": str(year),
            "month": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12"
            ],
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "daily_statistic": "daily_maximum",
            "time_zone": "utc+00:00",
            "frequency": "6_hourly",
            "area": [71.4, 179.9, 51.2, -129.9]
        }

        client = cdsapi.Client()
        client.retrieve(dataset, request, target = r"ERA5/Temperature Data/Alaska Raw Max Temp/{}.nc".format(year))

    def read_data(self):
        ds_max_1946 = nc.Dataset(r"ERA5/Temperature Data/Monthly/monthly_max_1940.nc")
        print(ds_max_1946)

        # jan_1_max_temp_1946 = ds_max_1946["t2m"][:, 45, 100] - 273.15

        # plt.plot([i for i in range(0, 365)], jan_1_max_temp_1946, label = "1946")
        # plt.legend()
        # plt.show()

    def download_all_data(self):
        for year in range(1943, 1944):
            time.sleep(10)
            self.download_data(year)
        
    def create_json(self):
        results_dict = {"coverage": "us-states-era5", "contains": [], "variable": "T2M", "data": {}}
        for year in range(1947, 2024):
            self.read_data(year)

    def generate_grayscale_colors(self, num_colors):
        grayscale_colors = []
        for i in range(num_colors):
            # Calculate the grayscale value, decreasing as i increases
            gray_value = int(255 * (1 - i / (num_colors - 1)**1.1))
            # Create the RGB color in grayscale
            color = f"rgb({gray_value}, {gray_value}, {gray_value})"
            grayscale_colors.append(color)
        return grayscale_colors

    # convert day of year to date
    def day_of_year_to_date(self, year: int, day_of_year: int) -> str:
        from datetime import datetime, timedelta
        """
        Converts a day of the year to a date in the format "yyyy-mm-dd".

        Parameters:
            year (int): The year (e.g., 2024).
            day_of_year (int): The day of the year (1-365 or 1-366 for leap years).

        Returns:
            str: The corresponding date in "yyyy-mm-dd" format.
        """
        try:
            # Calculate the date by adding the day_of_year to January 1st of the given year
            date = datetime(year, 1, 1) + timedelta(days=day_of_year)
            return date.strftime("%Y-%m-%d")
        except ValueError as e:
            return f"Error: {e}"
        
    def create_netcdf_file(self, file_name, time_data, t2m_data, lat_data, lon_data):
        """
        Creates a NetCDF4 file with variables: time, T2M, latitude, and longitude.

        Parameters:
        - file_name (str): The name of the NetCDF file to create.
        - time_data (list or numpy array): Array of datetime strings representing time.
        - t2m_data (numpy array): Array of shape (1, 96, 232) representing T2M values.
        - lat_data (numpy array): Array of shape (96,) representing latitudes.
        - lon_data (numpy array): Array of shape (232,) representing longitudes.
        """
        # Create a new NetCDF4 file
        ncfile = nc.Dataset(file_name, mode='w', format='NETCDF4')

        try:
            # Define dimensions
            time_dim = ncfile.createDimension('time', None)  # Unlimited dimension
            lat_dim = ncfile.createDimension('latitude', len(lat_data))
            lon_dim = ncfile.createDimension('longitude', len(lon_data))

            # Create variables with compression
            time_var = ncfile.createVariable('time', 'str', ('time',))
            lat_var = ncfile.createVariable('latitude', 'f4', ('latitude',))
            lon_var = ncfile.createVariable('longitude', 'f4', ('longitude',))
            t2m_var = ncfile.createVariable('T2M', 'f4', ('time', 'latitude', 'longitude'))

            # Add attributes (optional)
            time_var.units = 'datetime'
            lat_var.units = 'degrees_north'
            lon_var.units = 'degrees_east'
            t2m_var.units = 'Kelvin'

            time_var[:] = np.array(time_data)
            lat_var[:] = lat_data
            lon_var[:] = lon_data
            t2m_var[:, :] = t2m_data

        finally:
            # Close the NetCDF file
            ncfile.close()
            print(f"NetCDF file '{file_name}' created successfully.")

    def create_netcdf_files(self, state):
        files = sorted(os.listdir(r"ERA5/Temperature Data/{} Raw Max Temp".format(state)))
        for i, file in enumerate(files):
            if os.path.isfile(os.path.join(r"ERA5/Temperature Data/{} Raw Max Temp".format(state), file)):
                year = file.split(".")[0]
                ds = nc.Dataset(os.path.join(r"ERA5/Temperature Data/{} Raw Max Temp".format(state), file))
                days = len(ds["valid_time"][:])

                for i in range(days):
                    date = self.day_of_year_to_date(int(year), int(ds["valid_time"][i]))
                    max_daily_temp = ds["t2m"][i, :, :]
                    latitude = ds["latitude"][:]
                    longitude = ds["longitude"][:]

                    print(latitude.shape)
                    print(longitude.shape)
                    print(max_daily_temp.shape)

                    # write to new netcdf file
                    # self.create_netcdf_file(os.path.join(r"ERA5/Temperature Data/{} Clean Max Temp/{}".format(state, year), "{}.nc".format(date)), date, max_daily_temp, latitude, longitude)

    def points_in_state(self, lats, lons, state_abbreviation):
        geo_data = json.loads(open(r"/Users/kcox1729/Downloads/scdoshi us-geojson master geojson-state/{}.geojson".format(state_abbreviation), "r").read())["geometry"]
        state_shape = shape(geo_data)
        mask = np.zeros((len(lats), len(lons)))
        centroid = state_shape.centroid
        num_points = 0

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if lon > 180:
                    lon -= 360
                if state_shape.contains(Point(lon, lat)):
                    mask[i, j] = 1
                    num_points += 1

        return mask, num_points, centroid

    def get_month_and_day(self, date_string):
        from datetime import datetime as dt
        """
        Accepts a string in the format "yyyy-mm-dd" and returns
        the 3-letter abbreviation of the month and the day of the month.

        Parameters:
            date_string (str): Date in the format "yyyy-mm-dd"

        Returns:
            tuple: (month_abbreviation (str), day (int))
        """
        try:
            # Parse the input string into a datetime object
            date_object = dt.strptime(date_string, "%Y-%m-%d")
            
            # Extract the month abbreviation and day of the month
            month_abbreviation = date_object.strftime("%b")
            day = date_object.day
            
            return month_abbreviation, day
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")

    def process_state_data(self, state):
        if state == "Alaska":
            files = sorted(os.listdir(r"ERA5/Temperature Data/Alaska Raw Max Temp"))
            state_abbreviation = "AK"
        elif state == "Hawaii":
            files = sorted(os.listdir(r"ERA5/Temperature Data/Hawaii Raw Max Temp"))
            state_abbreviation = "HI"
        else:
            raise ValueError("State not supported")

        lat_lon_ds = nc.Dataset(os.path.join(r"ERA5/Temperature Data/{} Raw Max Temp".format(state), files[0]))
        lats = lat_lon_ds["latitude"][:]
        lons = lat_lon_ds["longitude"][:]
        mask, num_points, centroid = self.points_in_state(lats, lons, state_abbreviation)
        weights = np.cos(np.radians(lats))
        weights /= np.mean(weights)

        results_dict = {"results": {}, "num_points": num_points, "centroid": [centroid.x, centroid.y]}

        for file in files:
            if os.path.isfile(os.path.join(r"ERA5/Temperature Data/{} Raw Max Temp".format(state), file)):
                year = file.split(".")[0]
                ds = nc.Dataset(os.path.join(r"ERA5/Temperature Data/{} Raw Max Temp".format(state), file))
                days = ds["valid_time"][:]

                results_dict["results"][year] = {}

                for i in days:
                    date = self.day_of_year_to_date(int(year), int(ds["valid_time"][i]))
                    month, day = self.get_month_and_day(date)

                    max_daily_temp = ds["t2m"][i, :, :]
                    masked_temp = np.where(mask == 1, max_daily_temp, 0)
                    weighted_temp = masked_temp * weights[:, np.newaxis]
                    nonzero_temps = weighted_temp[np.nonzero(weighted_temp)]
                    average_daily_temp = np.mean(nonzero_temps) - 273.15

                    if month not in results_dict["results"][year]:
                        results_dict["results"][year][month] = []

                    results_dict["results"][year][month] += [average_daily_temp]

            print("finished year", year)

        existing_results = json.load(open(r"ERA5/Temperature Data/JSON Files/us-states-era5-t2m.json", "r"))
        existing_results["data"][state_abbreviation] = results_dict
        existing_results["contains"].append(state_abbreviation)
        json.dump(existing_results, open(r"ERA5/Temperature Data/JSON Files/us-states-era5-t2m.json", "w"))

    def global_timeseries_average(self):
        pass

    #         result["contains"].append(full_name)
    #         result["data"][full_name] = {}
    #         result["data"][full_name]["results"] = results_dict
    #         result["data"][full_name]["num_points"] = num_points
    #         result["data"][full_name]["centroid"] = [centroid.x, centroid.y]

# bounds
# southeast: 41.5, -71.85; lat = 263, lon = 173
# southwest: 41.5, -76.18; lat = 263, lon = 166
# northeast: 44.6, -71.85; lat = 269, lon = 173
# northwest: 44.6, -76.18; lat = 269, lon = 166

# latitude formula
# -90+0.5j
# longitutde formula
# -180+0.625i
# for lat in range(263, 270):
#     for lon in range(166, 174):
#         actual_lat_coord = -90+0.5*lat
#         actual_lon_coord = -180+0.625*lon
#         move_relevant_data_to_csv("./MERRA2/Temperature Data/Clean Max Temp/", "./MERRA2/Temperature Data/Max Temp/", str(actual_lat_coord) + "_" + str(actual_lon_coord), lat, lon)

# problem dates: 04-29-08, 05-31-15

if __name__ == "__main__":
    # RetrieveSingleVariable(password = "9x&HA$+pM%C)M2N", data_location = "daily/M2SDNXSLV.5.12.4", output_path = "MERRA2/Temperature Data/Min Temp", var = "T2MMIN").download_data()
    # data = GlobalAverageByDay().make_global_average_dict()
    # print(data)
    # polygon_data = json.load(open(r"Regional Geojsons/newengland.geojson", "r"))
    # CollectRegionalDataFromApp(polygon_data = polygon_data["features"][0]["geometry"]).aggregate_inside_points_temp_data()
    # for region in ["southatlanticgulfregion", "tennesseeregion", "texasgulfregion", "uppercoloradoregion", "uppermississippiregion"]:
    #     CollectRegionalData(polygon_data = r"Regional Geojsons/{}.geojson".format(region)).aggregate_inside_points_temp_data()
    # result = json.loads(open(r"ERA5/Temperature Data/JSON Files/us-states-era5-t2m.json", "r").read())
    result = {"contains": [], "data": {}, "coverage": "us-states-regions", "variable": "T2MMIN", "years": [i for i in range(1980, 2023)]}
    directory = os.listdir(r"/Users/kcox1729/Downloads/scdoshi us-geojson master geojson-state")
    for i, state in enumerate(directory):
        full_name = state.split(".")[0]
        file_name = full_name.lower().strip().replace(" ", "")

        # first create regional geojson and place it in folder
        geo_data = json.loads(open(r"/Users/kcox1729/Downloads/scdoshi us-geojson master geojson-state/{}".format(state), "r").read())["geometry"]

        # compute centroid of geo_data
        centroid = shape(geo_data).centroid

        # then run analysis
        lat_points = Dataset(r"MERRA2/Temperature Data/Min Temp/1980/1980-01-01.nc")["lat"][:]
        lon_points = Dataset(r"MERRA2/Temperature Data/Min Temp/1980/1980-01-01.nc")["lon"][:]
        results_dict, num_points = CollectRegionalData(polygon_data = geo_data, data_directory = r"MERRA2/Temperature Data/Min Temp", var = "T2MMIN", from_app = True, name = file_name, write_json = False).aggregate_inside_points_temp_data(lat_points, lon_points)
        if results_dict is None:
            continue
        else:
            result["contains"].append(full_name)
            result["data"][full_name] = {}
            result["data"][full_name]["results"] = results_dict
            result["data"][full_name]["num_points"] = num_points
            result["data"][full_name]["centroid"] = [centroid.x, centroid.y]

        print(full_name, "finshed", i)

    json.dump(result, open(r"MERRA2/JSON Files/Regional Aggregates/us-states-regions-t2m-min.json", "w"))

    # # fig = CountyLevelData().make_comparison_plot("MA", "1980")
    # # fig.show()

    # # print(CountyLevelData().concordance_by_month_year("MA"))

    # # eppa_data = EPPARegionAggregation()
    # # eppa_data.aggregate_inside_points_temp_data()

    # RetrieveSingleVariable(password = "9x&HA$+pM%C)M2N", data_location = "daily/M2T1NXSLV.5.12.4/", output_path = "./MERRA2/Specific Humidity/", var = "QV2M").download_data()

    # era5_data = ERA5Data()
    # era5_data.process_state_data("Alaska")
    # for year in range(1940, 2024):
    #     era5_data.download_data(year)
    # era5_data.read_data()
    # era5_data.download_all_data()
    # era5_data = ERA5Data()
    # era5_data.process_state_data("Alaska")
    # era5_data.create_netcdf_files("Hawaii")
