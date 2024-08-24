import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import netCDF4 as nc
from graphing_utils import TimeSeriesForOneRegion
import statsmodels.api as sm
from plotly.subplots import make_subplots
import json
import scipy.stats as sps

class FirstPass:
    def __init__(self, data_path = r"MERRA2/Temperature Data/Clean Max Temp/Boston.csv") -> None:
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.years = [i for i in range(1980, 2023)]

    def count_extreme_days(self, year, cutoff = 35):
        return np.where(self.df[year] > cutoff, 1, 0).sum()

    def simple_temperature_anomaly_correlation(self):
        count_dict = {}
        for year in self.years:
            num_extreme_days = self.count_extreme_days(str(year))
            count_dict[year] = int(num_extreme_days)

        return count_dict

    def plot_extreme_days(self, path_to_external_anamoly_data = r"temperature_anamoly.csv"):
        fig = go.Figure()
        anamoly_df = pd.read_csv(path_to_external_anamoly_data)
        anamoly_values = anamoly_df[(anamoly_df["Year"] > 1979) & (anamoly_df["Year"] < 2023)]["No_Smoothing"]
        extremes_values = list(self.simple_temperature_anomaly_correlation().values())
        fig.add_trace(go.Scatter(x = anamoly_values, y = extremes_values, name = "Boston", mode = "markers"))
        fig.update_layout(xaxis = dict(title = "Temperature Anamoly"), yaxis = dict(title = "# Days Above 35C"))
        fig.show()

class MachinesLearning:
    def __init__(self, path_to_train_dataset = r"MERRA2/Temperature Data/Clean Max Temp/combined_data.nc", path_to_test_dataset = r"MERRA2/Temperature Data/Clean Max Temp/combined_data_test.nc"):
        self.path_to_train_dataset = path_to_train_dataset
        self.path_to_test_dataset = path_to_test_dataset
        # set train and test period - will depend on the size of the dataset
        self.input_train_days = 1827 - 366
        self.output_train_days = 366

    def get_training_data(self):
        # Open the NetCDF file
        dataset = nc.Dataset(self.path_to_train_dataset, mode='r')
        temperature_data = dataset.variables['maxdailytemperature2m'][:]
        
        # Prepare the training and testing sequences
        X_train = temperature_data[:self.input_train_days]  # Input: First four years
        y_train = temperature_data[self.input_train_days:self.input_train_days + self.output_train_days]  # Output: Fifth year

        # Reshape the data for the CNN-LSTM model
        X_train = X_train.reshape((1, self.input_train_days, 361, 576, 1))  # Add batch and channel dimensions
        y_train = y_train.reshape((1, self.output_train_days, 361, 576, 1))  # Add batch and channel dimensions

        return X_train, y_train
    
    def get_test_data(self):
        dataset = nc.Dataset(self.path_to_test_dataset, mode='r')
        temperature_data = dataset.variables['maxdailytemperature2m'][:]
        
        X_test = temperature_data[:self.input_train_days]
        y_test = temperature_data[self.input_train_days:self.input_train_days + self.output_train_days]

        X_test = X_test.reshape((1, self.input_train_days, 361, 576, 1))  # Add batch and channel dimensions
        y_test = y_test.reshape((1, self.output_train_days, 361, 576, 1))  # Add batch and channel dimensions

        return X_test, y_test
    
    def train_model(self):
        X_train, y_train = self.get_training_data()

        # Define the input shape
        input_shape = (self.input_train_days, 361, 576, 1)

        # Define the input layer
        main_input = Input(shape=input_shape, name='main_input')

        # Reshape input to (train_days, 361, 576, 1) for the CNN
        x = TimeDistributed(Reshape((361, 576, 1)))(main_input)

        # Convolutional layers to capture spatial features
        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
        x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
        x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)

        # Flatten the spatial dimensions
        x = TimeDistributed(Flatten())(x)

        # Reshape to (train_days, -1) for the LSTM
        x = Reshape((self.input_train_days, -1))(x)

        # LSTM layers to capture temporal dependencies
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64)(x)

        # Fully connected layers for final prediction
        x = Dense(32, activation='relu')(x)
        output = TimeDistributed(Dense(1))(x)

        # Create the model
        model = Model(inputs=main_input, outputs=output)
        print("active 2")

        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Summary of the model
        model.summary()

        # Training
        history = model.fit(X_train, y_train, epochs=50, batch_size=1, validation_split=0.2)

        X_test, y_test = self.get_test_data()
        evaluation = model.evaluate(X_test, y_test)
        print(evaluation)
        print(f"Training Loss: {evaluation[0]}")
        print(f"Training MAE: {evaluation[1]}")

    def create_dataset(self, data, n_lookback=30):
        X, Y = [], []
        for i in range(n_lookback, len(data)):
            X.append(data[i - n_lookback:i])
            Y.append(data[i])
        return np.array(X), np.array(Y)

    def build_boston_model(self):
        wide_df = pd.read_csv("MERRA2/Temperature Data/Clean Max Temp/Boston.csv").drop(columns = ["Unnamed: 0"])
        flattened_df_train = wide_df[wide_df.columns[1:-2]].values.flatten(order = "F")
        flattened_df_test = wide_df[wide_df.columns[-2]].values.flatten(order = "F")

        # Assuming `temps` is your array of daily temperatures from 1980 to 2022
        train_data = flattened_df_train[:len(flattened_df_train)-730]  # up to end of 2020
        val_data = flattened_df_train[len(flattened_df_train)-730:len(flattened_df_train)-365]  # 2021 for validation
        test_data = flattened_df_train[len(flattened_df_train)-365:]  # 2022 for testing

        X_train, Y_train = self.create_dataset(train_data)
        X_val, Y_val = self.create_dataset(val_data)
        X_test, Y_test = self.create_dataset(test_data)

        # Reshape for LSTM input
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Model configuration
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(30, 1)),  # 30 days history, 1 feature (temperature)
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Fit the model
        model.fit(X_train, Y_train, epochs=50, validation_data=(X_val, Y_val))
        val_mse = model.evaluate(X_val, Y_val)
        print(f"Validation MSE: {val_mse}")

        # Generate predictions for the validation and test set
        predictions_test = model.predict(X_test)

        # Plot test predictions against actual data
        plt.plot(Y_test, label='Actual Temperatures', color='blue')
        plt.plot(predictions_test, label='Predicted Temperatures', color='red')
        plt.title('Test Data - 2022')
        plt.xlabel('Day')
        plt.ylabel('Temperature')
        plt.legend()

        plt.show()

class RegionalPrediction:
    def __init__(self, year, day_of_year, data_directory = r"MERRA2/Temperature Data/Clean Max Temp"):
        self.year = year
        self.day_of_year = day_of_year
        self.data_directory = data_directory

    def make_dataframe(self):
        files = os.listdir(self.data_directory)
        lats = []
        lons = []
        max_temps = []
        for file in files:
            if re.match("^[0-9]", file):
                df = pd.read_csv(os.path.join(self.data_directory, file))
                lat, lon = file[:-4].split("_")
                lat, lon = float(lat), float(lon)
                lats.append(lat)
                lons.append(lon)
                max_temps.append(df[str(self.year)].loc[self.day_of_year])

        df = pd.DataFrame({"Lat": lats, "Lon": lons, "Max Temp": max_temps})
        
        return df
    
    def prepare_data(self):
        data = self.make_dataframe()

        # Normalize the data
        self.scaler = StandardScaler()
        data[['Lat', 'Lon', 'Max Temp']] = self.scaler.fit_transform(data[['Lat', 'Lon', 'Max Temp']])

        # Split data into training and test sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Prepare training input vector
        X_train = train_data.drop(columns=['Max Temp']).values
        y_train = train_data['Max Temp'].values

        # Prepare testing input vector (latitude and longitude only)
        X_test = test_data.drop(columns=['Max Temp']).values
        y_test = test_data['Max Temp'].values

        return X_train, y_train, X_test, y_test

    def train_model(self):
        X_train, y_train, X_test, y_test = self.prepare_data()

        input_dim = X_train.shape[1]

        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)  # Output layer for temperature prediction
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()

        # Train the model
        history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

        predictions = model.predict(X_test)
        print(self.scaler.inverse_transform(np.concatenate((X_test, predictions), axis = 1)))
        print(self.scaler.inverse_transform(np.concatenate((X_test, y_test.reshape((23, 1))), axis = 1)))

        # Evaluate the model
        mse = model.evaluate(X_test, y_test)
        print(f"Mean Squared Error: {mse}")

class TemporalPrediction:
    def __init__(self, data_directory = r"MERRA2/Temperature Data/Clean Max Temp/Boston.csv"):
        self.data_directory = data_directory

    def create_dataset(self, data, n_lookback=30):
        X, Y = [], []
        for i in range(n_lookback, len(data)):
            X.append(data[i - n_lookback:i])
            Y.append(data[i])
        return np.array(X), np.array(Y)

    def train_model(self):
        wide_df = pd.read_csv(self.data_directory).drop(columns = ["Unnamed: 0"])
        flattened_df = wide_df[wide_df.columns[1:]].values.flatten(order = "F")
        all_days = [i for i in range(len(flattened_df))]

        # Assuming `temps` is your array of daily temperatures from 1980 to 2022
        train_data = flattened_df[:len(flattened_df)-730]  # up to end of 2020
        val_data = flattened_df[len(flattened_df)-730:len(flattened_df)-365]  # 2021 for validation
        test_data = flattened_df[len(flattened_df)-365:]  # 2022 for testing

        plt.plot(all_days[:len(train_data)], train_data, color = "green", label = "Training Set")
        plt.plot(all_days[len(train_data):len(train_data)+len(val_data)], val_data, color = "blue", label = "Validation Set")
        plt.plot(all_days[len(train_data)+len(val_data):], test_data, color = "red", label = "Testing Set")
        plt.legend()
        plt.show()

        X_train, Y_train = self.create_dataset(train_data)
        X_val, Y_val = self.create_dataset(val_data)
        X_test, Y_test = self.create_dataset(test_data)

        # Reshape for LSTM input
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Model configuration
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(30, 1)),  # 30 days history, 1 feature (temperature)
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Fit the model
        model.fit(X_train, Y_train, epochs=50, validation_data=(X_val, Y_val))
        val_mse = model.evaluate(X_val, Y_val)
        print(f"Validation MSE: {val_mse}")

        # Generate predictions for the validation and test set
        predictions_test = model.predict(X_test)

        # Plot test predictions against actual data
        plt.plot(Y_test, label='Actual Temperatures', color='blue')
        plt.plot(predictions_test, label='Predicted Temperatures', color='red')
        plt.title('Test Data - 2022')
        plt.xlabel('Day')
        plt.ylabel('Temperature')
        plt.legend()
        plt.show()

class SimpleDeltaMax:
    def __init__(self, data_directory = r"MERRA2/Temperature Data/Clean Max Temp", location = (42.0, -71.875)):
        self.data_directory = data_directory
        self.location = location
        self.file_name = f"{self.location[0]}_{self.location[1]}.csv"
        self.years = [i for i in range(1980, 2023)]

    def build_dataset(self):
        df = pd.read_csv(os.path.join(self.data_directory, self.file_name)).drop(columns = ["Unnamed: 0"])
        for col in df.columns:
            print(len(df[col]))

class MonthlyMaxTempPatterns:
    def __init__(self, lat, lon, file = None, data_directory = r"MERRA2/Temperature Data"):
        self.data_directory = data_directory
        self.lat = lat
        self.lon = lon
        self.file = file
        self.file_name = f"{self.lat}_{self.lon}.csv"

    def analyze_max_temps_for_one_month(self, month, metric: ["Avg", "Max", "Min", "Std", "Median"]):
        statistics_df = TimeSeriesForOneRegion(self.lat, self.lon, data_directory = self.data_directory, file = self.file).generate_statistics_df()
        month_df = statistics_df[statistics_df["Month"] == month]
        
        X = month_df["Year"]
        y = month_df[metric]
        
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        coef = results.params[1]
        p_value = results.pvalues[1]

        return coef, p_value

    def make_regression_table(self, title):
        fig = make_subplots(rows = 2, cols = 3, specs = [[{"type": "table"}, {"type": "table"}, {"type": "table"}], [{"type": "table"}, {"type": "table"}, {"type": "table"}]],
                            subplot_titles = ["Avg", "Max", "Min", "Std", "Median", ""], vertical_spacing = 0.05)
        for i, metric in enumerate(["Avg", "Max", "Min", "Std", "Median"]):
            metric_df = pd.DataFrame()
            coefs = []
            p_values = []
            for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
                coef, p_value = self.analyze_max_temps_for_one_month(month, metric)
                coefs.append(coef)
                p_values.append(p_value)
                
            metric_df["Month"] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            metric_df["Color"] = ["rgb(177, 156, 217)" if i > 0.05 else "rgb(75, 0, 130)" for i in p_values]
            metric_df["Coefficient"] = [round(i, 4) for i in coefs]
            metric_df["P-Value"] = [round(i, 4) for i in p_values]

            fig.add_trace(go.Table(
                    header = dict(
                        values = ["Month", "Coefficient", "P-Value"],
                        fill_color = "paleturquoise",
                        font = dict(color = "black", size = 12),
                        align = "left"
                    ),
                    cells = dict(
                        values = [metric_df["Month"], metric_df["Coefficient"], metric_df["P-Value"]],
                        fill_color = [["rgb(155, 206, 200)"]*12, ["rgb(155, 206, 200)"]*12, metric_df["Color"]],
                        font = dict(color = "white", size = 12),
                        align = "center"
                    )
                ),
                row = i//3 + 1,
                col = i%3 + 1
            )

        fig.update_layout(
            height = 800, width = 1200
        )
        fig.add_annotation(x = 0.95, y = 0.25, text = "P-Value < 0.05: Dark Purple<br>P-Value > 0.05: Light Purple", showarrow = False, font = dict(color = "black", size = 16))

        fig.update_layout(title = title)
        fig.show()

class GlobalMonthlyTempFromGlobalAverageTemp:
    def __init__(self, var = "T2MMEAN"):
        self.global_averages = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/global_average_{}.json".format(var.lower())))
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def compile_data(self):
        monthly_data_dict = {}
        for month in self.months:
            this_month_df = pd.DataFrame()
            this_month_data = []
            for year in range(1980, 2023):
                if year == 1980:
                    prior_year_monthly_data = self.global_averages[str(year)][month]
                    prior_year_monthly_average_temp = sum(prior_year_monthly_data)/len(prior_year_monthly_data)
                else:
                    this_year_temp_data = self.global_averages[str(year)][month]
                    this_year_monthly_average_temp = sum(this_year_temp_data)/len(this_year_temp_data)
                    monthly_diff = this_year_monthly_average_temp - prior_year_monthly_average_temp
                    prior_year_monthly_data = this_year_monthly_average_temp

                    this_month_data.append(monthly_diff)
            
            this_month_df["Year"] = [year for year in range(1981, 2023)]
            this_month_df["Values"] = this_month_data
            monthly_data_dict[month] = this_month_df

        return monthly_data_dict
    
    def train_model(self):
        from sklearn.linear_model import LinearRegression

        df = pd.DataFrame()
        for month in self.months:
            X = self.compile_data()[month]["Year"].values.reshape(-1, 1)
            y = self.compile_data()[month]["Values"].values

            model = LinearRegression()
            model.fit(X, y)
            # Predict future values
            y_pred = model.predict(X)

            coef = model.coef_[0]
            intercept = model.intercept_

            df = pd.concat([df, pd.DataFrame({"Month": [month], "Coef": [coef], "Intercept": [intercept]})], ignore_index = True)
            df.to_csv(r"global_yearly_to_monthly_coefs.csv")

        return df

class RegressionModel:
    def __init__(self, path_to_data, from_app = False, scenario = "ct", show_year_graph = False, end_year = 2050):
        # the data (for the region of interest) should be in the form of a json file with the following structure:
        # {
        #     "1980": {
        #         "Jan": [...],
        #         "Feb": [...],
        #         ...
        #     },
        #     "1981": {
        #         "Jan": [...],
        #         "Feb": [...],
        #         ...
        #     }
        #     ...
        # }
        self.show_year_graph = show_year_graph
        self.end_year = end_year
        self.years_for_prediction = [i for i in range(2023, end_year + 1)]
        self.global_averages = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/global_average_t2mmean.json"))
        # the region can either be an actual region of a longitude/latitude pair
        if from_app:
            self.regional_averages = path_to_data # expects path_to_data to be a json already
        else:
            self.regional_averages = json.load(open(path_to_data))

        self.scenario = scenario
        if scenario == "aa":
            self.scenario_predictions = pd.read_csv(r"aa_t2m.csv")
            self.mesm_predictions = self.scenario_predictions[(self.scenario_predictions["Year"] < self.end_year + 1) & (self.scenario_predictions["Year"] > 2022)].values[:, 1:]
            self.x_axis_median = np.percentile(self.mesm_predictions, 50, axis = 1)
        elif scenario == "ct":
            self.scenario_predictions = pd.read_csv(r"ct_t2m.csv")
            self.mesm_predictions = self.scenario_predictions[(self.scenario_predictions["Year"] < self.end_year + 1) & (self.scenario_predictions["Year"] > 2022)].values[:, 1:]
            self.x_axis_median = np.percentile(self.mesm_predictions, 50, axis = 1)

        self.global_average_temp_by_year_df = pd.read_csv(r"global_average_temp_by_year.csv")

    def global_local_regression(self, historical_X, historical_y):
        from scipy.stats import linregress

        regression_result = linregress(historical_X, historical_y)

        return regression_result.slope, regression_result.intercept, regression_result.rvalue, regression_result.pvalue, regression_result.stderr, regression_result.intercept_stderr

    def make_bound(self, upper_percentile, lower_percentile, regression_data, future_data):
        slope, intercept, r_value, p_value, std_coef, std_intercept = regression_data

        slope_samples = np.random.normal(slope, std_coef, 400)
        intercept_samples = np.random.normal(intercept, std_intercept, 400)
        predicted_samples = slope_samples*future_data + intercept_samples

        upper_bound = np.percentile(predicted_samples, upper_percentile, axis = 1)
        median = np.percentile(predicted_samples, 50, axis = 1)
        lower_bound = np.percentile(predicted_samples, lower_percentile, axis = 1)

        return upper_bound, median, lower_bound

    def average_monthly_max_temp_regression(self):
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        by_temp_fig = make_subplots(rows = 3, cols = 4, subplot_titles = months, vertical_spacing = 0.1, shared_xaxes = True)
        if self.show_year_graph:
            by_year_fig = make_subplots(rows = 3, cols = 4, subplot_titles = months, vertical_spacing = 0.1, shared_xaxes = True)
        colors = ['#00429d', '#3a4198', '#533f94', '#673e90', '#783b8b', '#883887', '#963582', '#a4307e', '#b12a79', '#be2375', '#ca1770', '#d6006c']

        # construct y (local temperatures)
        years = [str(i) for i in range(1980, 2023)]
        for i, month in enumerate(months):
            y = []
            for year in years:
                regional_averages_list = self.regional_averages[year][month]
                y.append(sum(regional_averages_list)/len(regional_averages_list))

            X_historical = self.global_average_temp_by_year_df["Average"].values
            y = np.array(y)

            # transform data first to improve uncertainty, then do a regression of global average temp against regional average daily max temp
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_historical.reshape(-1, 1))
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
            slope, intercept, r_value, p_value, std_coef, std_intercept = self.global_local_regression(X_scaled.reshape(-1), y_scaled.reshape(-1))

            # construct historical regression line
            y_pred_historical = slope*X_scaled + intercept
            y_pred_historical = scaler_y.inverse_transform(y_pred_historical).reshape(-1)

            # scale manually becausing doing it via the scaler is giving me too many issues
            scaled_future_predictions = (self.mesm_predictions - scaler_X.mean_) / scaler_X.scale_

            # traces for predicted data
            y_future_upper, y_future_median, y_future_lower = self.make_bound(95, 5, (slope, intercept, r_value, p_value, std_coef, std_intercept), scaled_future_predictions)
            y_future_upper = scaler_y.inverse_transform(y_future_upper.reshape(-1, 1)).reshape(-1)
            y_future_median = scaler_y.inverse_transform(y_future_median.reshape(-1, 1)).reshape(-1)
            y_future_lower = scaler_y.inverse_transform(y_future_lower.reshape(-1, 1)).reshape(-1)

            by_temp_fig.add_trace(go.Scatter(x = X_historical, y = y, name = month, mode = "markers", marker = dict(color = colors[i])), row = i//4 + 1, col = i%4 + 1)
            by_temp_fig.add_trace(go.Scatter(x = X_historical, y = y_pred_historical, mode = "lines", showlegend = False, marker = dict(color = colors[i])), row = i//4 + 1, col = i%4 + 1)
            by_temp_fig.add_trace(go.Scatter(x = self.x_axis_median, y = y_future_median, mode = "lines", showlegend = False, marker = dict(color = "orange")), row = i//4 + 1, col = i%4 + 1)
            by_temp_fig.add_trace(go.Scatter(x = self.x_axis_median, y = y_future_upper, fill=None, mode='lines', line=dict(color='#FFD580', width = 0.1), showlegend=False), row = i//4 + 1, col = i%4 + 1)
            by_temp_fig.add_trace(go.Scatter(x = self.x_axis_median, y = y_future_lower, fill='tonexty', mode='lines', line=dict(color='#FFD580', width = 0.1), showlegend=False), row = i//4 + 1, col = i%4 + 1)
            by_temp_fig.update_xaxes(title = f"p-value = {p_value:.4f}", row = i//4 + 1, col = i%4 + 1)

            if self.show_year_graph:
                by_year_fig.add_trace(go.Scatter(x = years, y = y, name = month, mode = "markers", marker = dict(color = colors[i])), row = i//4 + 1, col = i%4 + 1)
                by_year_fig.add_trace(go.Scatter(x = years, y = y_pred_historical, mode = "lines", showlegend = False, marker = dict(color = colors[i])), row = i//4 + 1, col = i%4 + 1)
                by_year_fig.add_trace(go.Scatter(x = self.years_for_prediction, y = y_future_median, mode = "lines", showlegend = False, marker = dict(color = "orange")), row = i//4 + 1, col = i%4 + 1)
                by_year_fig.add_trace(go.Scatter(x = self.years_for_prediction, y = y_future_upper, fill=None, mode='lines', line=dict(color='#FFD580', width = 0.1), showlegend=False), row = i//4 + 1, col = i%4 + 1)
                by_year_fig.add_trace(go.Scatter(x = self.years_for_prediction, y = y_future_lower, fill='tonexty', mode='lines', line=dict(color='#FFD580', width = 0.1), showlegend=False), row = i//4 + 1, col = i%4 + 1)

                by_year_fig.update_xaxes(tick0 = 1980, dtick=10, row=i//4 + 1, col=i%4 + 1)

        by_temp_fig.update_layout(height = 650, width = 910, title = "Average Monthly Max Temperature Regression - By Global Yearly Average Mean Temperature")
        by_temp_fig.update_yaxes(title = "Regional Monthly Average Max Temperature (C)", row = 2, col = 1)
        by_temp_fig.add_annotation(text = "Global Yearly Average Max Temperature (C)", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

        by_year_fig.update_layout(height = 650, width = 910, title = "Average Monthly Max Temperature Regression - By Year")
        by_year_fig.update_yaxes(title = "Regional Monthly Average Max Temperature (C)", row = 2, col = 1)
        by_year_fig.add_annotation(text = "Year", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

        return by_temp_fig, by_year_fig
    
    def average_monthly_max_temp_regression_diff(self):
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        by_temp_fig = make_subplots(rows = 3, cols = 4, subplot_titles = months, vertical_spacing = 0.1, shared_xaxes = True)
        if self.show_year_graph:
            by_year_fig = make_subplots(rows = 3, cols = 4, subplot_titles = months, vertical_spacing = 0.1, shared_xaxes = True)
        colors = ['#00429d', '#3a4198', '#533f94', '#673e90', '#783b8b', '#883887', '#963582', '#a4307e', '#b12a79', '#be2375', '#ca1770', '#d6006c']

        # construct y (local temperatures)
        years = [str(i) for i in range(1980, 2023)]
        for i, month in enumerate(months):
            y = []
            for year in years:
                regional_averages_list = self.regional_averages[year][month]
                y.append(sum(regional_averages_list)/len(regional_averages_list))

            X_historical = self.global_average_temp_by_year_df["Average"].values
            y = np.array(y)

            # transform data first to improve uncertainty, then do a regression of global average temp against regional average daily max temp
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_historical.reshape(-1, 1))
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
            slope, intercept, r_value, p_value, std_coef, std_intercept = self.global_local_regression(X_scaled.reshape(-1), y_scaled.reshape(-1))

            # construct historical regression line
            y_pred_historical = slope*X_scaled + intercept
            y_pred_historical = scaler_y.inverse_transform(y_pred_historical).reshape(-1)

            # get data from both ct and aa scenarios
            scenario_predictions_aa = pd.read_csv(r"aa_t2m.csv")
            scenario_predictions_ct = pd.read_csv(r"ct_t2m.csv")

            mesm_predictions_aa = scenario_predictions_aa[(scenario_predictions_aa["Year"] < self.end_year + 1) & (scenario_predictions_aa["Year"] > 2022)].values[:, 1:]
            mesm_predictions_ct = scenario_predictions_ct[(scenario_predictions_ct["Year"] < self.end_year + 1) & (scenario_predictions_ct["Year"] > 2022)].values[:, 1:]

            x_axis_median_aa = np.percentile(mesm_predictions_aa, 50, axis = 1)
            x_axis_median_ct = np.percentile(mesm_predictions_ct, 50, axis = 1)

            scaled_future_predictions_aa = scaler_X.transform(x_axis_median_aa.reshape(-1, 1))
            scaled_future_predictions_ct = scaler_X.transform(x_axis_median_ct.reshape(-1, 1))

            # traces for predicted data
            y_future_upper_aa, y_future_median_aa, y_future_lower_aa = self.make_bound(95, 5, (slope, intercept, r_value, p_value, std_coef, std_intercept), scaled_future_predictions_aa)
            y_future_upper_aa = scaler_y.inverse_transform(y_future_upper_aa.reshape(-1, 1)).reshape(-1)
            y_future_median_aa = scaler_y.inverse_transform(y_future_median_aa.reshape(-1, 1)).reshape(-1)
            y_future_lower_aa = scaler_y.inverse_transform(y_future_lower_aa.reshape(-1, 1)).reshape(-1)

            y_future_upper_ct, y_future_median_ct, y_future_lower_ct = self.make_bound(95, 5, (slope, intercept, r_value, p_value, std_coef, std_intercept), scaled_future_predictions_ct)
            y_future_upper_ct = scaler_y.inverse_transform(y_future_upper_ct.reshape(-1, 1)).reshape(-1)
            y_future_median_ct = scaler_y.inverse_transform(y_future_median_ct.reshape(-1, 1)).reshape(-1)
            y_future_lower_ct = scaler_y.inverse_transform(y_future_lower_ct.reshape(-1, 1)).reshape(-1)

            # get the differences - this uses a different estimation procedure than just subtracting the two sets of predictions
            # it is a little inefficient, but it is good enough
            # estimate the variance
            var_est_aa = ((y_future_upper_aa - y_future_median_aa)/1.645)**2
            var_est_ct = ((y_future_upper_ct - y_future_median_ct)/1.645)**2
            
            diff_est_median = y_future_median_aa - y_future_median_ct
            dist = sps.norm(diff_est_median, np.sqrt(var_est_aa + var_est_ct))
            diff_est_upper = dist.ppf(0.95)
            diff_est_lower = dist.ppf(0.05)

            # only for first trace do we add the legend, so that there aren't a million items in the legend box
            if i == 0:
                # aa
                by_temp_fig.add_trace(go.Scatter(x = x_axis_median_aa, y = y_future_median_aa, mode = "lines", name = "Prediction Acc. Actions", legendgroup = "aa", marker = dict(color = "orange")), row = i//4 + 1, col = i%4 + 1)
                by_temp_fig.add_trace(go.Scatter(x = x_axis_median_aa, y = y_future_upper_aa, fill=None, mode='lines', line=dict(color='#FFD580', width = 0.1), showlegend = False, legendgroup = "aa"), row = i//4 + 1, col = i%4 + 1)
                by_temp_fig.add_trace(go.Scatter(x = x_axis_median_aa, y = y_future_lower_aa, fill='tonexty', mode='lines', line=dict(color='#FFD580', width = 0.1), showlegend = False, legendgroup = "aa"), row = i//4 + 1, col = i%4 + 1)

                # ct
                by_temp_fig.add_trace(go.Scatter(x = x_axis_median_ct, y = y_future_median_ct, mode = "lines", name = "Prediction Current Trends", legendgroup = "ct", marker = dict(color = "#0BBBD6")), row = i//4 + 1, col = i%4 + 1)
                by_temp_fig.add_trace(go.Scatter(x = x_axis_median_ct, y = y_future_upper_ct, fill=None, mode='lines', line=dict(color='#52BBCB', width = 0.1), showlegend = False, legendgroup = "ct"), row = i//4 + 1, col = i%4 + 1)
                by_temp_fig.add_trace(go.Scatter(x = x_axis_median_ct, y = y_future_lower_ct, fill='tonexty', mode='lines', line=dict(color='#52BBCB', width = 0.1), showlegend = False, legendgroup = "ct"), row = i//4 + 1, col = i%4 + 1)
            else:
                by_temp_fig.add_trace(go.Scatter(x = x_axis_median_aa, y = y_future_median_aa, mode = "lines", showlegend = False, legendgroup = "aa", marker = dict(color = "orange")), row = i//4 + 1, col = i%4 + 1)
                by_temp_fig.add_trace(go.Scatter(x = x_axis_median_aa, y = y_future_upper_aa, fill=None, mode='lines', line=dict(color='#FFD580', width = 0.1), showlegend = False, legendgroup = "aa"), row = i//4 + 1, col = i%4 + 1)
                by_temp_fig.add_trace(go.Scatter(x = x_axis_median_aa, y = y_future_lower_aa, fill='tonexty', mode='lines', line=dict(color='#FFD580', width = 0.1), showlegend = False, legendgroup = "aa"), row = i//4 + 1, col = i%4 + 1)

                by_temp_fig.add_trace(go.Scatter(x = x_axis_median_ct, y = y_future_median_ct, mode = "lines", showlegend = False, legendgroup = "ct", marker = dict(color = "#0BBBD6")), row = i//4 + 1, col = i%4 + 1)
                by_temp_fig.add_trace(go.Scatter(x = x_axis_median_ct, y = y_future_upper_ct, fill=None, mode='lines', line=dict(color='#52BBCB', width = 0.1), showlegend = False, legendgroup = "ct"), row = i//4 + 1, col = i%4 + 1)
                by_temp_fig.add_trace(go.Scatter(x = x_axis_median_ct, y = y_future_lower_ct, fill='tonexty', mode='lines', line=dict(color='#52BBCB', width = 0.1), showlegend = False, legendgroup = "ct"), row = i//4 + 1, col = i%4 + 1)

            by_temp_fig.add_trace(go.Scatter(x = X_historical, y = y, name = month, mode = "markers", marker = dict(color = colors[i])), row = i//4 + 1, col = i%4 + 1)
            by_temp_fig.add_trace(go.Scatter(x = X_historical, y = y_pred_historical, mode = "lines", showlegend = False, marker = dict(color = colors[i])), row = i//4 + 1, col = i%4 + 1)
            by_temp_fig.update_xaxes(title = f"p-value = {p_value:.4f}", row = i//4 + 1, col = i%4 + 1)

            # plot difference in by year graph
            if self.show_year_graph:
                # historical data not applicable here, only predictions
                if i == 0:
                    by_year_fig.add_trace(go.Scatter(x = self.years_for_prediction, y = diff_est_median, mode = "lines", name = "Difference From Current Trends (Median)", legendgroup = "diff", marker = dict(color = "orange")), row = i//4 + 1, col = i%4 + 1)
                    by_year_fig.add_trace(go.Scatter(x = self.years_for_prediction, y = diff_est_upper, fill=None, mode='lines', line=dict(color='#FFD580', width = 0.1), showlegend=False, legendgroup = "diff"), row = i//4 + 1, col = i%4 + 1)
                    by_year_fig.add_trace(go.Scatter(x = self.years_for_prediction, y = diff_est_lower, fill='tonexty', mode='lines', line=dict(color='#FFD580', width = 0.1), showlegend=False, legendgroup = "diff"), row = i//4 + 1, col = i%4 + 1)

                else:
                    by_year_fig.add_trace(go.Scatter(x = self.years_for_prediction, y = diff_est_median, mode = "lines", showlegend = False, legendgroup = "diff", marker = dict(color = "orange")), row = i//4 + 1, col = i%4 + 1)
                    by_year_fig.add_trace(go.Scatter(x = self.years_for_prediction, y = diff_est_upper, fill=None, mode='lines', line=dict(color='#FFD580', width = 0.1), showlegend=False, legendgroup = "diff"), row = i//4 + 1, col = i%4 + 1)
                    by_year_fig.add_trace(go.Scatter(x = self.years_for_prediction, y = diff_est_lower, fill='tonexty', mode='lines', line=dict(color='#FFD580', width = 0.1), showlegend=False, legendgroup = "diff"), row = i//4 + 1, col = i%4 + 1)

                by_year_fig.update_xaxes(tick0 = 1980, dtick=10, row=i//4 + 1, col=i%4 + 1)

        by_temp_fig.update_layout(height = 650, width = 910, title = "Average Monthly Max Temperature Regression - By Global Yearly Average Mean Temperature")
        by_temp_fig.update_yaxes(title = "Regional Monthly Average Max Temperature (C)", row = 2, col = 1)
        by_temp_fig.add_annotation(text = "Global Yearly Average Max Temperature (C)", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

        if self.show_year_graph:
            by_year_fig.update_layout(height = 650, width = 910, title = "Accelerated Actions Difference From Current Trends Baseline")
            by_year_fig.update_yaxes(title = "Difference in Average Max Daily Temperature (C)", row = 2, col = 1)
            by_year_fig.add_annotation(text = "Year", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

        if self.show_year_graph:
            return by_temp_fig, by_year_fig
        else:
            return by_temp_fig

    def main(self):
        if self.scenario == "diff":
            fig1, fig2 = self.average_monthly_max_temp_regression_diff()
        else:
            fig1, fig2 = self.average_monthly_max_temp_regression()

        return fig1, fig2

class HeatWavePrediction:
    # working definition of a heat wave:
    # a period of 5 or more consecutive days with a max daily temperature of exceeding the 
    # historical average (1980-1990 for this dataset) by 5 C
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.data = json.load(open(self.data_directory))
        self.years = [str(i) for i in range(1980, 2023)]

    def daily_t_max(self):
        daily_t_max_dict = {}
        for y, year in enumerate(self.years):
            for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
                temps_in_this_month = self.data[year][month]
                for i, temp in enumerate(temps_in_this_month):
                    day = month + "-" + str(i+1)
                    if daily_t_max_dict.get(day):
                        # we've already seen this day, so update the average using the average filter
                        # need to check if this is a leap year first
                        if day == "Feb-29":
                            y = y/4
                        daily_t_max_dict[day] = (y*daily_t_max_dict[day] + temp)/(y+1)
                    else:
                        # first time seeing this day
                        daily_t_max_dict[day] = temp

        return daily_t_max_dict

    def count_heat_waves_by_year(self):
        daily_t_max_dict = self.daily_t_max()
        heat_waves_by_year_dict = {}
        current_heat_wave_len = 0
        for year in self.years:
            for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
                monthly_data = self.data[year][month]
                for i, temp in enumerate(monthly_data):
                    day = month + "-" + str(i+1)
                    if temp > daily_t_max_dict[day] + 5:
                        # heat wave condition met
                        current_heat_wave_len += 1
                    else:
                        # current heat wave is over, add and reset
                        # can't do this above because then we would add 1 to the year's heat wave count
                        # for every day that was in the heat wave
                        if current_heat_wave_len > 4:
                            heat_waves_by_year_dict[year] = heat_waves_by_year_dict.get(year, 0) + 1
                        current_heat_wave_len = 0
            if not heat_waves_by_year_dict.get(year):
                heat_waves_by_year_dict[year] = 0

        return heat_waves_by_year_dict

    def num_heat_waves_per_year(self):
        fig = go.Figure()
        heat_waves_by_year_dict = self.count_heat_waves_by_year()
        heat_waves_by_year_list = [heat_waves_by_year_dict[year] for year in self.years]
        
        fig.add_trace(go.Scatter(x = self.years, y = heat_waves_by_year_list, name = "Heat Waves", mode = "lines", line = dict(color = "red")))
        fig.show()

    def global_average_temp(self):
        global_averages = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/global_average_t2mmax.json"))
        global_average_dict = {}
        for year in self.years:
            for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
                global_average_dict[year][month] = sum(global_averages[year][month])/len(global_averages[year][month])

        return global_average_dict

class DownscaleGlobalAverageTemp:
    def __init__(self):
        self.global_averages = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/global_average_t2mmean.json"))

    def downscale(self):
        pass

if __name__ == "__main__":
    # MonthlyMaxTempPatterns(42.5, -71.875, file = r"Regional Averages/global_averages.json").make_regression_table("Regression Table for World")
    # HeatWavePrediction(r"MERRA2/JSON Files/Coordinates/data_41_-73.75_t2mmax.json").num_heat_waves_per_year()
    fig1, fig2 = RegressionModel(r"MERRA2/JSON Files/Regional Aggregates/areaweightednewengland_average_t2mmax.json", show_year_graph = True, scenario = "ct").main()
    fig1.show()
    fig2.show()
    # df = GlobalMonthlyTempFromGlobalAverageTemp().train_model()
