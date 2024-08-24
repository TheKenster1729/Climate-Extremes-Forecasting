import paramiko
from scp import SCPClient
import os
from netCDF4 import Dataset
import netCDF4 as nc
import pandas as  pd
import datetime
import re
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import json
from shapely.geometry import Point, shape
import xarray as xr

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
    def __init__(self, polygon_data, data_directory = r"MERRA2/Temperature Data/Max Temp", from_app = False):
        self.from_app = from_app
        if not from_app:
            self.polygon_file_name = os.path.basename(polygon_data)

        if self.from_app:
            self.polygon_data = polygon_data
        else:
            assert type(polygon_data) == str # if this is not coming from the app, expect a file path
            self.polygon_data = json.load(open(polygon_data, "r"))["features"][0]["geometry"]
        self.data_directory = data_directory
        self.lat_points = np.linspace(start = -90, stop = 90, num = 361)
        self.lon_points = np.linspace(start = -180, stop = 179.375, num = 576)
        self.years = [str(i) for i in range(1980, 2023)]

        # Extract coordinates
        self.polygon = shape(self.polygon_data)

    def get_data(self, file_path):
        dataset = Dataset(file_path)
        values = dataset.RangeBeginningDate, dataset["lon"][:], dataset["lat"][:], dataset["T2MMAX"][0, :, :]
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

        # just need to create weights once
        weights = []
        for lat, lon in inside_points:
            weights.append(np.cos(np.radians(lat)))
        weights = np.array(weights)
        weights /= np.array(weights).mean()

        # iterate through each year in the data set
        for year in self.years:
            results_dict[year] = {}
            # each file corresponds to one day
            for file in os.listdir(os.path.join(self.data_directory, year)):
                time, df = self.make_dataframe(os.path.join(self.data_directory, year, file))
                _, month, day = time.split("-")
                # iterate through each point in the region and store it in the list
                this_day_in_region = []
                for lat, lon in inside_points:
                    this_day_in_region.append(df.loc[lat, lon] - 273.15)
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

        if not self.from_app: # place the results in the regional aggregates folder
            json.dump(results_dict, open(r"MERRA2/JSON Files/Regional Aggregates/{}".format(self.polygon_file_name[:-8] + "_" + "average" + "_" + "t2mmax" + ".json"), "w"))

        return results_dict

class GlobalAverageByDay:
    def __init__(self, folder_of_files = r"MERRA2/Temperature Data/Max Temp") -> None:
        self.data_folder = folder_of_files
        self.years = [i for i in range(1980, 2023)]

    def get_data(self, file_path):
        dataset = Dataset(file_path)

        return dataset.RangeBeginningDate, dataset["lon"][:], dataset["lat"][:], dataset["T2MMAX"][0, :, :]
    
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
        for year in self.years[0:1]:
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
        # sorted_data_dict = {}
        # months = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}
        # for year in data.keys():
        #     data[year] = sorted(data[year].items(), key=lambda x: x[0])
        #     sorted_data_dict[year] = {months[i]: j for i, j in data[year]}
        #     for month in sorted_data_dict[year]:
        #         sorted_data_dict[year][month] = sorted(sorted_data_dict[year][month].items(), key=lambda x: x[0])
        #         sorted_data_dict[year][month] = [i[1] for i in sorted_data_dict[year][month]]

        #     print(f"Finished year {year}")

        # json.dump(sorted_data_dict, open(r"Regional Averages/global_averages.json", "w"))

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
    # data = GlobalAverageByDay().make_global_average_dict()
    # print(data)
    # polygon_data = json.load(open(r"Regional Geojsons/newengland.geojson", "r"))
    # CollectRegionalDataFromApp(polygon_data = polygon_data["features"][0]["geometry"]).aggregate_inside_points_temp_data()
    for region in ["southatlanticgulfregion", "tennesseeregion", "texasgulfregion", "uppercoloradoregion", "uppermississippiregion"]:
        CollectRegionalData(polygon_data = r"Regional Geojsons/{}.geojson".format(region)).aggregate_inside_points_temp_data()