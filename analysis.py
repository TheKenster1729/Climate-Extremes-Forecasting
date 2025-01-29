import pandas as pd
import numpy as np
import os
import re
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import netCDF4 as nc
import json
import plotly.express as px

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

    def compare_scenarios(self):
        ct_scenarios = pd.read_csv(r"ct_t2m.csv")
        aa_scenarios = pd.read_csv(r"aa_t2m.csv")

        ct_scenarios = ct_scenarios[(ct_scenarios["Year"] < self.end_year + 1) & (ct_scenarios["Year"] > 2022)].values[:, 1:]
        aa_scenarios = aa_scenarios[(aa_scenarios["Year"] < self.end_year + 1) & (aa_scenarios["Year"] > 2022)].values[:, 1:]

        ct_median = np.percentile(ct_scenarios, 50, axis = 1)
        ct_upper = np.percentile(ct_scenarios, 95, axis = 1)
        ct_lower = np.percentile(ct_scenarios, 5, axis = 1)

        aa_median = np.percentile(aa_scenarios, 50, axis = 1)
        aa_upper = np.percentile(aa_scenarios, 95, axis = 1)
        aa_lower = np.percentile(aa_scenarios, 5, axis = 1)

        x = np.arange(2023, self.end_year + 1)

        # Define colors for the shaded areas
        color_set1 = 'rgba(128, 221, 255, 0.2)'  # Light blue
        color_set2 = 'rgba(187, 128, 255, 0.2)'  # Light purple

        # Create the plot
        fig = go.Figure()

        # Add shaded area for Set 1
        fig.add_traces([
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([ct_upper, ct_lower[::-1]]),
                fill='toself',
                fillcolor=color_set1,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend = False
            ),
            go.Scatter(
                x=x,
                y=ct_median,
                mode='lines',
                line=dict(color='#80ddff', width = 2),
                name='Current Trends'
            )
        ])

        # Add shaded area for Set 2
        fig.add_traces([
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([aa_upper, aa_lower[::-1]]),
                fill='toself',
                fillcolor=color_set2,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend = False
            ),
            go.Scatter(
                x=x,
                y=aa_median,
                mode='lines',
                line=dict(color='#bb80ff', width = 2),
                name='Accelerated Actions'
            )
        ])

        # Customize layout
        fig.update_layout(
            title="Accelerated Actions vs Current Trends Mean Global Temperature Projections<br>2023-2050</br>",
            xaxis_title="Year",
            yaxis_title="Global Mean Temperature (°C)",
            template="plotly_white",
            showlegend=True,
            height = 600,
            width = 1000
        )


        return fig

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
        from sklearn.preprocessing import StandardScaler
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

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
        by_temp_fig.update_yaxes(title = "Regional Monthly Average Max Temperature (°C)", row = 2, col = 1)
        by_temp_fig.add_annotation(text = "Global Yearly Average Max Temperature (°C)", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

        by_year_fig.update_layout(height = 650, width = 910, title = "Average Monthly Max Temperature Regression - By Year")
        by_year_fig.update_yaxes(title = "Regional Monthly Average Max Temperature (°C)", row = 2, col = 1)
        by_year_fig.add_annotation(text = "Year", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

        return by_temp_fig, by_year_fig
    
    def average_monthly_max_temp_regression_diff(self):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        from sklearn.preprocessing import StandardScaler
        import scipy.stats as sps

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
        by_temp_fig.update_yaxes(title = "Regional Monthly Average Max Temperature (°C)", row = 2, col = 1)
        by_temp_fig.add_annotation(text = "Global Yearly Average Max Temperature (°C)", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))
        by_temp_fig.update_yaxes(title = "Regional Monthly Average Max Temperature (°C)", row = 2, col = 1)
        by_temp_fig.add_annotation(text = "Global Yearly Average Max Temperature (°C)", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

        if self.show_year_graph:
            by_year_fig.update_layout(height = 650, width = 910, title = "Accelerated Actions Difference From Current Trends Baseline")
            by_year_fig.update_yaxes(title = "Difference in Average Max Daily Temperature (°C)", row = 2, col = 1)
            by_year_fig.update_yaxes(title = "Difference in Average Max Daily Temperature (°C)", row = 2, col = 1)
            by_year_fig.add_annotation(text = "Year", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

            return by_temp_fig, by_year_fig
        else:
            return by_temp_fig

    def main(self):
        if self.scenario == "diff":
            fig1, fig2 = self.average_monthly_max_temp_regression_diff()
        else:
            fig1, fig2 = self.average_monthly_max_temp_regression()

        return fig1, fig2

class AppFunctions:
    def __init__(self, dataset_name = "MERRA2", var = "T2MMAX", timeframe = "merra2", end_year = 2050):
        self.dataset_name = dataset_name
        self.var = var
        self.end_year = end_year
        self.path_to_regression_results = f"Regression Results/{self.dataset_name}/regression_results-{self.dataset_name}-{self.var}.csv"
        self.timeframe = timeframe
        self.years = [str(i) for i in range(1980, 2023)] if self.timeframe == "merra2" else [str(i) for i in range(1940, 2024)]
        self.full_years = [str(i) for i in range(1980, self.end_year + 1)] if self.timeframe == "merra2" else [str(i) for i in range(1940, self.end_year + 1)]
        self.projections_years = [str(i) for i in range(2023, self.end_year + 1)]
        if self.timeframe == "merra2" and self.dataset_name == "ERA5":
            self.path_to_regression_results = f"Regression Results/ERA5/regression_results-{self.dataset_name}-{self.var}-merra2_timeframe.csv"
        self.regression_results = pd.read_csv(self.path_to_regression_results)
        if self.dataset_name == "ERA5":
            self.data = json.load(open(r"ERA5/Temperature Data/JSON Files/us-states-era5-t2m.json"))
        else:
            if self.var == "T2MMAX":
                self.data = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/us-states-regions.json"))
            elif self.var == "T2MMIN":
                self.data = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/us-states-regions-t2m-min.json"))
            elif self.var == "T2MMEAN":
                self.data = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/us-states-regions-t2m-mean.json"))
            else:
                raise ValueError

        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        self.standardize_historical_global_temp()

    def make_slope_samples(self, region, month):
        beta_hat = self.regression_results[self.regression_results["Region"] == region][self.regression_results["Month"] == month]["Slope"].values[0]
        intercept_hat = self.regression_results[self.regression_results["Region"] == region][self.regression_results["Month"] == month]["Intercept"].values[0]
        beta_std = self.regression_results[self.regression_results["Region"] == region][self.regression_results["Month"] == month]["Std Coef"].values[0]
        intercept_std = self.regression_results[self.regression_results["Region"] == region][self.regression_results["Month"] == month]["Std Intercept"].values[0]

        beta_samples = np.random.normal(beta_hat, beta_std, 400)
        intercept_samples = np.random.normal(intercept_hat, intercept_std, 400)

        return beta_samples, intercept_samples

    def get_mean_global_temp(self):
        if self.dataset_name == "MERRA2":
            global_averages = pd.read_csv(r"global_average_temp_by_year.csv")
            mean_temps = global_averages["Average"].values
        elif self.dataset_name == "ERA5":
            global_averages = json.load(open(r"ERA5/Temperature Data/JSON Files/world-average.json"))
            mean_temps = []
            for entry in global_averages:
                if entry["name"] in self.years:
                    temps_list = entry["data"]
                    temps_list = [temp for temp in temps_list if temp is not None]
                    mean_temps.append(sum(temps_list)/len(temps_list))
            mean_temps = np.array(mean_temps)

        return mean_temps

    def standardize_historical_global_temp(self):
        mean_temps = self.get_mean_global_temp()
        scaler_mean = self.regression_results["Scaler X Mean"].values[0]
        scaler_scale = self.regression_results["Scaler X Scale"].values[0]
        self.mean_temps_scaled = (mean_temps - scaler_mean)/scaler_scale

    def standardize_future_global_temp(self, x):
        scaler_mean = self.regression_results["Scaler X Mean"].values[0]
        scaler_scale = self.regression_results["Scaler X Scale"].values[0]
        x_scaled = (x - scaler_mean)/scaler_scale

        return x_scaled

    def get_regional_temp_data(self, region, month):
        y = []
        for year in self.years:
            regional_averages_list = self.data["data"][region]["results"][str(year)][month]
            y.append(sum(regional_averages_list)/len(regional_averages_list))

        y = np.array(y)

        return y

    def get_regional_temp_data_by_month(self, region):
        regional_data = {}
        for month in self.months:
            regional_data[month] = {}
            y = self.get_regional_temp_data(region, month)
            regional_data[month]["y"] = y

        return regional_data

    def generate_historical_bounds(self, region, month):
        x = self.mean_temps_scaled[:, np.newaxis]
        beta_samples, intercept_samples = self.make_slope_samples(region, month)
        y_pred_historical = beta_samples*x + intercept_samples

        median = np.median(y_pred_historical, axis = 1)
        lower = np.percentile(y_pred_historical, 5, axis = 1)
        upper = np.percentile(y_pred_historical, 95, axis = 1)

        return median, lower, upper
    
    def get_future_global_temp(self, scenario):
        if scenario == "aa":
            future_global_temp = pd.read_csv(r"aa_t2m.csv")
        elif scenario == "ct":
            future_global_temp = pd.read_csv(r"ct_t2m.csv")

        future_global_temp_filtered = future_global_temp[(future_global_temp["Year"] < self.end_year + 1) & (future_global_temp["Year"] > 2022)]
        future_global_temp_vals = future_global_temp_filtered.values[:, 1:]

        return future_global_temp_vals
    
    def generate_future_bounds(self, region, month, scenario):
        x = self.get_future_global_temp(scenario)
        x = self.standardize_future_global_temp(x)
        beta_samples, intercept_samples = self.make_slope_samples(region, month)
        y_pred_future = beta_samples*x + intercept_samples

        median = np.median(y_pred_future, axis = 1)
        lower = np.percentile(y_pred_future, 5, axis = 1)
        upper = np.percentile(y_pred_future, 95, axis = 1)

        return median, lower, upper

    def original_coordinates_historical(self, region, month):
        median, lower, upper = self.generate_historical_bounds(region, month)
        scaler_mean = self.regression_results[self.regression_results["Region"] == region][self.regression_results["Month"] == month]["Scaler Y Mean"].values[0]
        scaler_scale = self.regression_results[self.regression_results["Region"] == region][self.regression_results["Month"] == month]["Scaler Y Scale"].values[0]
        median_original = median*scaler_scale + scaler_mean
        lower_original = lower*scaler_scale + scaler_mean
        upper_original = upper*scaler_scale + scaler_mean

        return median_original, lower_original, upper_original

    def original_coordinates_future(self, region, month, scenario):
        median, lower, upper = self.generate_future_bounds(region, month, scenario)
        scaler_mean = self.regression_results[self.regression_results["Region"] == region][self.regression_results["Month"] == month]["Scaler Y Mean"].values[0]
        scaler_scale = self.regression_results[self.regression_results["Region"] == region][self.regression_results["Month"] == month]["Scaler Y Scale"].values[0]
        median_original = median*scaler_scale + scaler_mean
        lower_original = lower*scaler_scale + scaler_mean
        upper_original = upper*scaler_scale + scaler_mean

        return median_original, lower_original, upper_original

    def make_temp_plot(self, region, scenario):
        from plotly.subplots import make_subplots

        x_data = self.get_mean_global_temp()
        x_projections = np.median(self.get_future_global_temp(scenario), axis = 1)
        fig = make_subplots(rows = 3, cols = 4,
                            subplot_titles = [month for month in self.months], shared_xaxes = True, vertical_spacing = 0.1)
        for i, month in enumerate(self.months):
            median_historical, lower_historical, upper_historical = self.original_coordinates_historical(region, month)
            scatterplot_data = self.get_regional_temp_data(region, month)

            median_future, lower_future, upper_future = self.original_coordinates_future(region, month, scenario)

            plotting_df = pd.DataFrame({"x": x_projections, "y": median_future, "fill_lower": lower_future, "fill_upper": upper_future})
            plotting_df = plotting_df.sort_values(by = "x")

            fig.add_trace(go.Scatter(x = x_data, y = scatterplot_data, name = "Historical Data", mode = "markers", marker = dict(color = "blue"), showlegend = False), row = i//4 + 1, col = i%4 + 1)
            fig.add_trace(go.Scatter(
                x=x_data,
                y=median_historical,
                mode='lines',
                line=dict(color='orange'),
                name='Median',
                showlegend = False
            ), row = i//4 + 1, col = i%4 + 1)
            fig.add_trace(go.Scatter(x = plotting_df["x"], y = plotting_df["y"], mode = "lines", line = dict(color = "rgba(255, 162, 128, 1)"), name = "Future Median", showlegend = False), row = i//4 + 1, col = i%4 + 1)
            fig.add_trace(go.Scatter(
                x = list(plotting_df["x"]) + list(plotting_df["x"][::-1]),
                y = list(plotting_df["fill_lower"]) + list(plotting_df["fill_upper"][::-1]),
                fill='toself',
                fillcolor='rgba(255, 162, 128, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
            ), row = i//4 + 1, col = i%4 + 1)

        fig.update_layout(
            title=f'Historical Data with Projections (By Temperature)',
            height = 500,
            width = 600,
            margin = dict(l = 100, r = 0, t = 75, b = 0)
        )

        fig.update_yaxes(title = "Regional Monthly-Averaged Daily Max Temperature (°C)", row = 2, col = 1)
        fig.add_annotation(text = "Global Yearly Mean Temperature (°C)", xref = "paper", yref = "paper", x = 0.5, y = -0.1, showarrow = False, font = dict(size = 14))

        return fig
    
    def make_year_plot(self, region, scenario):
        from plotly.subplots import make_subplots

        x_projections = np.median(self.get_future_global_temp(scenario), axis = 1)
        fig = make_subplots(rows = 3, cols = 4,
                            subplot_titles = [month for month in self.months], shared_xaxes = True, vertical_spacing = 0.1)
        for i, month in enumerate(self.months):
            median_historical, lower_historical, upper_historical = self.original_coordinates_historical(region, month)
            scatterplot_data = self.get_regional_temp_data(region, month)

            median_future, lower_future, upper_future = self.original_coordinates_future(region, month, scenario)

            plotting_df = pd.DataFrame({"x": x_projections, "y": median_future, "fill_lower": lower_future, "fill_upper": upper_future})
            plotting_df["year"] = self.projections_years
            plotting_df = plotting_df.sort_values(by = "x")

            fig.add_trace(go.Scatter(x = self.years, y = scatterplot_data, name = "Historical Data", mode = "markers", marker = dict(color = "blue"), showlegend = False), row = i//4 + 1, col = i%4 + 1)
            fig.add_trace(go.Scatter(
                x=self.years,
                y=median_historical,
                mode='lines',
                line=dict(color='orange'),
                name='Median',
                showlegend = False
            ), row = i//4 + 1, col = i%4 + 1)
            fig.add_trace(go.Scatter(x = plotting_df["year"], y = plotting_df["y"], mode = "lines", line = dict(color = "rgba(255, 162, 128, 1)"), name = "Future Median", showlegend = False), row = i//4 + 1, col = i%4 + 1)
            fig.add_trace(go.Scatter(
                x = list(plotting_df["year"]) + list(plotting_df["year"][::-1]),
                y = list(plotting_df["fill_lower"]) + list(plotting_df["fill_upper"][::-1]),
                fill='toself',
                fillcolor='rgba(255, 162, 128, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
            ), row = i//4 + 1, col = i%4 + 1)

        fig.update_layout(
            title=f'Historical Data with Projections (By Year)',
            height = 500,
            width = 600,
            margin = dict(l = 0, r = 0, t = 75, b = 0)
        )

        fig.update_yaxes(title = "Regional Monthly-Averaged Daily Max Temperature (°C)", row = 2, col = 1)
        fig.add_annotation(text = "Year", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

        return fig
    
    def make_plots(self, region, scenario):
        temp_fig = self.make_temp_plot(region, scenario)
        year_fig = self.make_year_plot(region, scenario)

        return temp_fig, year_fig

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

class CNNTempGridPrediction:
    def __init__(self):
        pass

    def main(self):
        import tensorflow as tf
        from tensorflow.keras import datasets, layers, models
        from netCDF4 import Dataset
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score

        year_train = 2021
        year_test = 2022

        train_data_location = "MERRA2/Temperature Data/Max Temp/" + str(year_train)
        test_data_location = "MERRA2/Temperature Data/Max Temp/" + str(year_test)
        train_files = os.listdir(train_data_location)
        test_files = os.listdir(test_data_location)

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        # training data
        for train_file, test_file in zip(train_files, test_files):
            data_train = Dataset(os.path.join(train_data_location, train_file))
            data_train = data_train.variables['T2MMAX'][0, :, :]

            this_day_X_train = data_train[260:280, 170:190]
            this_day_y_train = data_train[260:280, 190]

            X_train.append(this_day_X_train)
            y_train.append(this_day_y_train)

            data_test = Dataset(os.path.join(test_data_location, test_file))
            data_test = data_test.variables['T2MMAX'][0, :, :]

            this_day_X_test = data_test[260:280, 170:190]
            this_day_y_test = data_test[260:280, 190]

            X_test.append(this_day_X_test)
            y_test.append(this_day_y_test)

        X_train = np.stack(X_train)
        y_train = np.stack(y_train)
        X_test = np.stack(X_test)
        y_test = np.stack(y_test)

        model = models.Sequential()
        model.add(layers.Conv2D(64, (2,2), input_shape = (20, 20, 1), activation = "relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2,2), activation = "relu"))
        model.add(layers.Flatten())
        model.add(layers.Dense(20))

        model.compile(optimizer = "adam", loss = "mse", metrics = ["mae"])
        model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test))
        y_pred = model.predict(X_test)

        print(r2_score(y_test, y_pred))

class VolcanoPlot:
    def __init__(self, scenario = "ct", for_region = "states"):
        self.scenario = scenario
        self.for_region = for_region
        self.naming_df = pd.read_csv(r"region_names.csv")
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        self.global_average_temp_by_year_df = pd.read_csv(r"global_average_temp_by_year.csv")
        
        if self.for_region == "states":
            self.regions = [i[1] for i in self.naming_df[self.naming_df["Type"] == "State"].values]

    def global_local_regression(self, historical_X, historical_y):
        from scipy.stats import linregress

        regression_result = linregress(historical_X, historical_y)

        return regression_result.slope, regression_result.intercept, regression_result.rvalue, regression_result.pvalue, regression_result.stderr, regression_result.intercept_stderr
    
    def construct_coef_and_p_value_df(self):
        all_data = []
        for state in self.regions:
            path_to_state_data = r"MERRA2/JSON Files/Regional Aggregates/" + state + "_average_t2mmax.json"
            state_data = json.load(open(path_to_state_data))

            # construct y (local temperatures)
            years = [str(i) for i in range(1980, 2023)]
            for i, month in enumerate(self.months):
                y = []
                for year in years:
                    regional_averages_list = state_data[year][month]
                    y.append(sum(regional_averages_list)/len(regional_averages_list))

                X_historical = self.global_average_temp_by_year_df["Average"].values
                y = np.array(y)

                # transform data first to improve uncertainty, then do a regression of global average temp against regional average daily max temp
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_scaled = scaler_X.fit_transform(X_historical.reshape(-1, 1))
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
                slope, intercept, r_value, p_value, std_coef, std_intercept = self.global_local_regression(X_historical.reshape(-1), y.reshape(-1))
                slope_unscaled = scaler_y.inverse_transform(np.array([[slope]]))[0][0]/scaler_X.transform(np.array([[1]]))[0][0]
                intercept_unscaled = scaler_y.inverse_transform(np.array([[intercept]]))[0][0]

                all_data.append([state, month, slope_unscaled, intercept_unscaled, r_value, p_value, std_coef, std_intercept])

        df = pd.DataFrame(all_data, columns = ["Region", "Month", "Slope", "Intercept", "R Value", "P-Value", "Coefficient SD", "Intercept SD"])

        return df

    def create_volcano_plot(self):
        df = self.construct_coef_and_p_value_df()

        seasons = {"Winter": ["Dec", "Jan", "Feb"], "Spring": ["Mar", "Apr", "May"], "Summer": ["Jun", "Jul", "Aug"], "Fall": ["Sep", "Oct", "Nov"]}
        df["Season"] = df["Month"].apply(lambda x: next((k for k, v in seasons.items() if x in v), "Unknown"))
        df["-log10(P-Value)"] = -np.log10(df["P-Value"])

        fig = make_subplots(rows = 2, cols = 2, subplot_titles = ["Winter", "Spring", "Summer", "Fall"], specs = [[{"type": "scatter"}, {"type": "scatter"}], [{"type": "scatter"}, {"type": "scatter"}]])
        for i, season in enumerate(seasons):
            this_season_df = df[df["Season"] == season].reset_index(drop = True)

            fig.add_trace(go.Scatter(
                x = this_season_df["Slope"],
                y = this_season_df["-log10(P-Value)"],
                mode = "markers",
                marker = dict(color = "firebrick", size = 5),
                name = season,
                text = this_season_df["Region"]
            ), row = i//2 + 1, col = i%2 + 1)
            fig.update_xaxes(title = "Slope", row = i//2 + 1, col = i%2 + 1)
            fig.update_yaxes(title = "-log10(P-Value)", row = i//2 + 1, col = i%2 + 1)
            fig.add_annotation(text = "Num significant > 2: " + str(len(this_season_df[(this_season_df["P-Value"] < 0.05) & (this_season_df["Slope"] > 2)])), x = max(this_season_df["Slope"]) + 0.01, y = 0, showarrow = False, font = dict(size = 14),
                               row = i//2 + 1, col = i%2 + 1, align = "right")

        fig.update_layout(height = 800, width = 1200, title = "Volcano Plot of Slope and P-Value for Each Region", showlegend = False)
        
        return fig

class Vulcanalyzation:
    def __init__(self, path_to_json = r"MERRA2/JSON Files/Regional Aggregates/us-states-regions.json", regression = "center-and-scale"):
        self.data = json.load(open(path_to_json))
        self.regression = regression
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        self.seasons = {"Winter": ["Dec", "Jan", "Feb"], "Spring": ["Mar", "Apr", "May"], "Summer": ["Jun", "Jul", "Aug"], "Fall": ["Sep", "Oct", "Nov"]}
        self.global_average_temp_by_year_df = pd.read_csv(r"global_average_temp_by_year.csv")
        self.X_historical = self.global_average_temp_by_year_df["Average"].values

    def center_and_scale(self, historical_X, historical_y):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(historical_X.reshape(-1, 1))
        y_scaled = self.scaler_y.fit_transform(historical_y.reshape(-1, 1))

        return X_scaled, y_scaled
    
    def build_df(self, regression_type = None):
        if regression_type:
            self.regression = regression_type

        all_data = []
        for region in self.data["contains"]:
            num_points = self.data["data"][region]["num_points"]
            lon_centroid = self.data["data"][region]["centroid"][0]
            lat_centroid = self.data["data"][region]["centroid"][1]

            # construct y (local temperatures)
            years = [str(i) for i in range(1940, 2023)]
            for i, month in enumerate(self.months):
                season = next((k for k, v in self.seasons.items() if month in v), "Unknown")
                y = []
                for year in years:
                    regional_averages_list = self.data["data"][region]["results"][year][month]
                    y.append(sum(regional_averages_list)/len(regional_averages_list))

                y = np.array(y)

                if self.regression == "center-and-scale":
                    X_scaled, y_scaled = self.center_and_scale(self.X_historical, y)

                    regression_result = linregress(X_scaled.reshape(-1), y_scaled.reshape(-1))
                    slope, intercept, r_value, p_value, std_coef, std_intercept = regression_result.slope, regression_result.intercept, regression_result.rvalue, regression_result.pvalue, regression_result.stderr, regression_result.intercept_stderr

                elif self.regression == "simple":
                    regression_result = linregress(self.X_historical, y)

                    slope, intercept, r_value, p_value, std_coef, std_intercept = regression_result.slope, regression_result.intercept, regression_result.rvalue, regression_result.pvalue, regression_result.stderr, regression_result.intercept_stderr

                elif self.regression == "shift":
                    X_shifted = self.X_historical - np.mean(self.X_historical)
                    y_shifted = y - np.mean(y)

                    regression_result = linregress(X_shifted, y_shifted)
                    slope, intercept, r_value, p_value, std_coef, std_intercept = regression_result.slope, regression_result.intercept, regression_result.rvalue, regression_result.pvalue, regression_result.stderr, regression_result.intercept_stderr

                elif self.regression == "center":
                    X_centered = self.X_historical - np.mean(self.X_historical)
                    y_centered = y - np.mean(y)

                    regression_result = linregress(X_centered, y_centered)
                    slope, intercept, r_value, p_value, std_coef, std_intercept = regression_result.slope, regression_result.intercept, regression_result.rvalue, regression_result.pvalue, regression_result.stderr, regression_result.intercept_stderr

                    upper_bound = intercept + 1.96*std_intercept
                    lower_bound = intercept - 1.96*std_intercept
                else:
                    raise ValueError("Invalid regression type")

                all_data.append([slope, intercept, p_value, region, num_points, lon_centroid, lat_centroid, sum(self.data["data"][region]["results"]["1980"][month])/len(self.data["data"][region]["results"]["1980"][month]), month, season])

        df = pd.DataFrame(all_data, columns = ["slope", "intercept", "p_value", "region", "num_points", "lon_centroid", "lat_centroid", "baseline_temp", "month", "season"])
        cmi_df = pd.read_csv(r"state_cmi.csv")
        df = df.join(cmi_df.set_index("abbreviation"), on = "region", how = "inner")
        qv2m_df = pd.read_csv(r"qv2m_df.csv")
        df = df.join(qv2m_df.set_index(["region", "month"]), on = ["region", "month"], how = "inner")
        potential_height_df = pd.read_csv(r"potential_height_df.csv")
        df = df.join(potential_height_df.set_index(["region", "month"]), on = ["region", "month"], how = "inner")

        return df

    def quadratic_regression(self, show = False):
        all_data = []
        for region in self.data["contains"]:
            num_points = self.data["data"][region]["num_points"]
            lon_centroid = self.data["data"][region]["centroid"][0]
            lat_centroid = self.data["data"][region]["centroid"][1]

            # construct y (local temperatures)
            years = [str(i) for i in range(1980, 2023)]
            for i, month in enumerate(self.months):
                season = next((k for k, v in self.seasons.items() if month in v), "Unknown")
                y = []
                for year in years:
                    regional_averages_list = self.data["data"][region]["results"][year][month]
                    y.append(sum(regional_averages_list)/len(regional_averages_list))

                y = np.array(y)

                X_scaled, y_scaled = self.center_and_scale(self.X_historical, y)

                a, b, c = np.polyfit(X_scaled.reshape(-1), y_scaled.reshape(-1), 2)
                all_data.append([a, b, c, region, num_points, lon_centroid, lat_centroid, sum(self.data["data"][region]["results"]["1980"][month])/len(self.data["data"][region]["results"]["1980"][month]), month, season])

        df = pd.DataFrame(all_data, columns = ["a", "b", "c", "region", "num_points", "lon_centroid", "lat_centroid", "baseline_temp", "month", "season"])
        print(df[(df["a"] > 0.3) | (df["a"] < -0.3)].sort_values(by = "a", ascending = False)[["a", "region", "month"]].to_html(index = False))
        if show:
            df_group = df.groupby("month").agg({"a": "mean", "b": "mean", "c": "mean"}).reset_index()
            df_group.sort_values(by = "month", inplace = True, key = lambda x: [self.months.index(i) for i in x])
            df_group.plot(x = "month", y = ["a", "b", "c"], kind = "bar", title = "a, b, c vs month")
            plt.show()

        return df

    def compare_regression_types(self):
        df_cs = self.build_df("center-and-scale")
        df_simple = self.build_df("simple")

        table_data = []
        for month in self.months:
            cs_slope = df_cs[df_cs["month"] == month].reset_index(drop = True)["slope"].values[0]
            simple_slope = df_simple[df_simple["month"] == month].reset_index(drop = True)["slope"].values[0]

            table_data.append([month, cs_slope, simple_slope])

        table_df = pd.DataFrame(table_data, columns = ["Month", "Center-and-Scale Slope", "Simple Regression Slope"])

        fig = go.Figure()
        fig.add_trace(go.Table(
            header = dict(values = ["Month", "Center-and-Scale Slope", "Simple Regression Slope"]),
            cells = dict(values = [table_df["Month"], table_df["Center-and-Scale Slope"], table_df["Simple Regression Slope"]])
        ))
        fig.show()
        
    def plot_regression_type(self, state, month, regression_type = None):
        df = self.build_df(regression_type)
        df_state = df[df["region"] == state].reset_index(drop = True)
        df_month = df_state[df_state["month"] == month].reset_index(drop = True)
        predicted_temps = df_month["slope"].values[0]*self.X_historical + df_month["intercept"].values[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = self.X_historical,
            y = predicted_temps,
            mode = "lines",
            name = "Predicted Temps"
        ))
        
        fig.show()
        
    def random_forest_model(self):
        # create label column
        df = self.build_df(self.regression)

        df["label"] = ((df["p_value"] < 0.05) & (df["slope"] > np.percentile(df["slope"], 70))).astype(int)
        preprocessor = ColumnTransformer(
            transformers = [
                ("cat", OneHotEncoder(), ["season"])
            ],
            remainder = "passthrough"
        )
        
        X = df[["num_points", "baseline_temp", "lon_centroid", "lat_centroid", "season", "cmi", "avg_qv2m", "avg_potential_height"]]
        y = df["label"]

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators = 100, max_depth = 5, class_weight = "balanced"))
        ])

        model.fit(X, y)

        # compute feature importance
        importances = model.named_steps["classifier"].feature_importances_
        feature_names = [i.split("__")[1] for i in model.named_steps["preprocessor"].get_feature_names_out()]
        sorted_idx = importances.argsort()[::-1]
        importances_list = [importances[i] for i in sorted_idx]
        feature_names_list = [feature_names[i] for i in sorted_idx]

        return importances_list, feature_names_list
    
    def trends_by_season(self):
        df = self.build_df(self.regression)
        df["trend"] = ((df["p_value"] < 0.05) & (df["slope"] > np.percentile(df["slope"], 70))).astype(int)
        df_season = df.groupby("season").sum().reset_index()
        print(df_season["trend"])

        fig = px.bar(df_season, x = "season", y = "trend", title = "Trends by Season")
        fig.update_layout(xaxis_title = "Season", yaxis_title = "# Significant Trends")
        fig.show()

    def overall_trend_distribution(self):
        df = self.build_df(self.regression)
        fig = px.histogram(df[(df["month"] == "May") | (df["month"] == "Feb")], x = "slope", title = "Trend Distribution for May and Feb", marginal = "box")
        fig.show()

    def trends_by_month(self):
        df = self.build_df(self.regression)
        df["trend"] = (df["p_value"] < 0.05).astype(int)
        df["color"] = df["trend"].apply(lambda x: "Significant Trend" if x == 1 else "NS")

        # create a choropleth map of the trends
        fig = px.choropleth(
            df,
            locations = "region",
            locationmode = "USA-states",
            color = "color",
            title = f"Trends by Month",
            scope = "usa",
            facet_col = "month",
            facet_col_wrap = 4,
            facet_col_spacing = 0,
            color_discrete_map = {
            "Significant Trend": "#6ae66a",
            "NS": "#a83277"
        }
        )
        fig.update_layout(width = 1500, height = 1000)
        for i, trace in enumerate(fig.data):
            # creative way to show legend for only two traces
            trace.showlegend = i == 0 or i == len(fig.data) - 1

        return fig

    def strength_of_trends(self):
        df = self.build_df(self.regression)
        
        fig = px.choropleth(
            df,
            locations = "region",
            locationmode = "USA-states",
            color = "slope",
            title = f"Strength of Trends by Month",
            scope = "usa",
            facet_col = "month",
            facet_col_wrap = 4,
            facet_col_spacing = 0,
            color_continuous_scale = ["green", "yellow", "red"]
        )
        fig.update_layout(width = 1500, height = 1000)

        return fig
    
    def spatial_trends_plot(self):
        # same as trends_by_month, but showing the magnitude of trends
        pass

    def cart_tree(self):
        df = self.build_df(self.regression)
        df["label"] = ((df["p_value"] < 0.05) & (df["slope"] > np.percentile(df["slope"], 70))).astype(int)
        df["binary_cmi"] = df["cmi"].map(lambda x: 0 if x < 0 else 1)

        preprocessor = ColumnTransformer(
            transformers = [
                ("season", OneHotEncoder(), ["season"]),
                ("binary_cmi", OneHotEncoder(), ["binary_cmi"])
            ],
            remainder = "passthrough"
        )

        X = df[["num_points", "baseline_temp", "lat_centroid", "season", "binary_cmi", "avg_qv2m", "avg_potential_height"]]
        y = df["label"]

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth = 3))
        ])
        model.fit(X, y)

        print(model.score(X, y))

        # export tree
        export_graphviz(model.named_steps["classifier"], out_file = "cart_tree.dot", feature_names = model.named_steps["preprocessor"].get_feature_names_out())
        with open("cart_tree.dot") as f:
            dot_graph = f.read()
        graphviz.Source(dot_graph).render("cart_tree", format = "png")

    def multiple_regression_model(self):
        df = self.build_df(self.regression)
        df["season"] = df["season"].map(lambda x: 0 if x == "Winter" else 1 if x == "Spring" else 2 if x == "Summer" else 3)

        X = df[["num_points", "baseline_temp", "lon_centroid", "lat_centroid", "cmi"]]
        y = df["slope"].values.reshape(-1, 1)

        model = Pipeline([
            ("regressor", LinearRegression())
        ])

        model.fit(X, y)

        print(model.score(X, y))

        print(model.named_steps["regressor"].coef_)
        
    def random_forest_model_by_month(self):

        fig = make_subplots(rows = 3, cols = 4,
                            subplot_titles = self.months)
        df = self.build_df(self.regression)
        for i, month in enumerate(self.months):
            # create label column
            month_df = df[df["month"] == month].reset_index(drop = True)
            month_df["label"] = ((month_df["p_value"] < 0.05) & (month_df["slope"] > np.percentile(month_df["slope"], 70))).astype(int)

            preprocessor = ColumnTransformer(
                transformers = [
                    ("cat", OneHotEncoder(), ["season"])
                ],
                remainder = "passthrough"
            )

            X = month_df[["num_points", "baseline_temp", "lon_centroid", "lat_centroid", "season", "cmi"]]
            y = month_df["label"]

            model = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators = 100, max_depth = 5, class_weight = "balanced"))
            ])

            model.fit(X, y)

            # compute feature importance
            importances = model.named_steps["classifier"].feature_importances_
            feature_names = [i.split("__")[1] for i in model.named_steps["preprocessor"].get_feature_names_out()]
            sorted_idx = importances.argsort()[::-1]
            importances_list = [importances[i] for i in sorted_idx]
            feature_names_list = [feature_names[i] for i in sorted_idx]

            fig.add_trace(go.Bar(
                x = importances_list,
                y = feature_names_list,
                name = month,
                orientation = "h"
            ), row = i//4 + 1, col = i%4 + 1)

        fig.update_layout(height = 800, width = 1600, title = "Feature Importance for Each Month", showlegend = False)

        return fig
    
    def feature_against_slope(self, feature):
        df = self.build_df(self.regression)
        print(df)

        if feature != "season":
            X = df[feature].values.reshape(-1, 1)
            y = df["slope"].values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)

            plt.scatter(X, y)
            plt.plot(X, model.predict(X), color = "red")

        else:
            fig, ax = plt.subplots()
            for season in self.seasons:
                this_season_df = df[df["season"] == season].reset_index(drop = True)
                median_slope = np.median(this_season_df["slope"].values)

                ax.bar(season, median_slope, label = season)

            ax.set_xlabel("Season")
            ax.set_ylabel("Median Slope")
            ax.legend()

        plt.show()

    def split_slope_calculation(self, show = False):
        all_data = []
        for region in self.data["contains"]:

            # construct y (local temperatures)
            years_1 = [str(i) for i in range(1980, 2001)]
            years_2 = [str(i) for i in range(2001, 2023)]
            for i, month in enumerate(self.months):
                y_1 = []
                y_2 = []
                for year in years_1:
                    regional_averages_list = self.data["data"][region]["results"][year][month]
                    y_1.append(sum(regional_averages_list)/len(regional_averages_list))

                for year in years_2:
                    regional_averages_list = self.data["data"][region]["results"][year][month]
                    y_2.append(sum(regional_averages_list)/len(regional_averages_list))

                y_1 = np.array(y_1)
                y_2 = np.array(y_2)

                X_scaled_1, y_scaled_1 = self.center_and_scale(self.global_average_temp_by_year_df[self.global_average_temp_by_year_df["Year"] <= 2000]["Average"].values, y_1)
                X_scaled_2, y_scaled_2 = self.center_and_scale(self.global_average_temp_by_year_df[self.global_average_temp_by_year_df["Year"] > 2000]["Average"].values, y_2)

                regression_result_1 = linregress(X_scaled_1.reshape(-1), y_scaled_1.reshape(-1))
                slope_1, intercept_1, r_value_1, p_value_1, std_coef_1, std_intercept_1 = regression_result_1.slope, regression_result_1.intercept, regression_result_1.rvalue, regression_result_1.pvalue, regression_result_1.stderr, regression_result_1.intercept_stderr

                regression_result_2 = linregress(X_scaled_2.reshape(-1), y_scaled_2.reshape(-1))
                slope_2, intercept_2, r_value_2, p_value_2, std_coef_2, std_intercept_2 = regression_result_2.slope, regression_result_2.intercept, regression_result_2.rvalue, regression_result_2.pvalue, regression_result_2.stderr, regression_result_2.intercept_stderr

                all_data.append([region, month, slope_1, slope_2])

        df = pd.DataFrame(all_data, columns = ["region", "month", "slope_1980-2000", "slope_2000-2023"])

        # kolmogorov-smirnov test
        ks_stat, ks_p_value = ks_2samp(df["slope_1980-2000"].values, df["slope_2000-2023"].values)
        print(ks_stat, ks_p_value)

        if show:
            fig = make_subplots(rows = 3, cols = 4, subplot_titles = self.months)
            for i, month in enumerate(self.months):
                month_df = df[df["month"] == month].reset_index(drop = True)
                fig.add_trace(go.Bar(
                    x = ["1980-2000", "2000-2023"],
                    y = [month_df["slope_1980-2000"].mean(), month_df["slope_2000-2023"].mean()],
                    name = month
                ), row = i//4 + 1, col = i%4 + 1)
                # add vertical lines at medians of both groups
                # fig.add_vline(x = month_df["slope_1980-2000"].mean(), line_width = 2, line_color = "orange")
                # fig.add_vline(x = month_df["slope_2000-2023"].mean(), line_width = 2, line_color = "skyblue")
                # fig.update_layout(title = f"Distribution of Slope by Time Period for {month}")
            fig.update_layout(height = 1200, width = 1600, title = "Average Slope by Time Period")
            fig.show()

    def analyze_quadratic_regression(self):
        df_regression = self.build_df(self.regression)
        df_quad = self.quadratic_regression()
        df = pd.merge(df_regression, df_quad, on = ["region", "month"], how = "inner", suffixes = ("", "_duplicate"))
        df = df.loc[:, ~df.columns.str.endswith('_duplicate')]

        print(df)

        # create random forest model and cart tree
        preprocessor = ColumnTransformer(
            transformers = [
                ("cat", OneHotEncoder(), ["season"])
            ],
            remainder = "passthrough"
        )

        X = df[["num_points", "baseline_temp", "lon_centroid", "lat_centroid", "season", "cmi", "avg_qv2m", "avg_potential_height", "slope"]]
        y = df["a"]

        random_forest_model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestRegressor(n_estimators = 100, max_depth = 5))
        ])

        random_forest_model.fit(X, y)

        # compute feature importance
        importances = random_forest_model.named_steps["classifier"].feature_importances_
        feature_names = [i.split("__")[1] for i in random_forest_model.named_steps["preprocessor"].get_feature_names_out()]
        sorted_idx = importances.argsort()[::-1]
        importances_list = [importances[i] for i in sorted_idx]
        feature_names_list = [feature_names[i] for i in sorted_idx]
        for i, j in zip(importances_list, feature_names_list):
            print("name: ", j, "importance: ", i)

        # cart tree
        cart_tree_model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", DecisionTreeRegressor(max_depth = 3))
        ])
        cart_tree_model.fit(X, y)
        print(cart_tree_model.score(X, y))
        export_graphviz(cart_tree_model.named_steps["regressor"], out_file = "cart_tree.dot", feature_names = cart_tree_model.named_steps["preprocessor"].get_feature_names_out())
        with open("cart_tree.dot") as f:
            dot_graph = f.read()
        graphviz.Source(dot_graph).render("cart_tree_quadratic_analysis", format = "png")

    def svm_model(self):
        df = self.build_df(self.regression)

        X = df[["num_points", "baseline_temp", "lon_centroid", "lat_centroid", "season", "cmi", "avg_qv2m", "avg_potential_height"]]
        y = df["slope"]

        preprocessor = ColumnTransformer(
            transformers = [
                ("cat", OneHotEncoder(), ["season"])
            ],
            remainder = "passthrough"
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", SVR(kernel = "linear"))
        ])

        model.fit(X, y)
        print(model.score(X, y))

        pd.Series(abs(model.named_steps["regressor"].coef_[0]), index = model.named_steps["preprocessor"].get_feature_names_out()).nlargest(5).plot(kind = "barh")
        plt.show()

    def overall_warming_year(self):
        df = self.build_df(self.regression)
        df_average_slope_by_month = df[["region", "month", "slope"]].groupby(["region", "month"]).mean().reset_index()
        month_weights = {"Jan": 31/365, "Feb": 28/365, "Mar": 31/365, "Apr": 30/365, "May": 31/365, "Jun": 30/365, "Jul": 31/365, "Aug": 31/365, "Sep": 30/365, "Oct": 31/365, "Nov": 30/365, "Dec": 31/365}
        df_average_slope_by_month["weighted_slope"] = df_average_slope_by_month["slope"] * df_average_slope_by_month["month"].map(month_weights)
        df_average_slope_by_year = df_average_slope_by_month.groupby("region").sum().reset_index()[["region", "weighted_slope"]].sort_values("weighted_slope", ascending = False)
        print(df_average_slope_by_year)

    def mann_kendall_test(self):
        all_data = []
        for region in self.data["contains"]:

            # construct y (local temperatures)
            years = [str(i) for i in range(1980, 2023)]
            for i, month in enumerate(self.months):
                y = []
                for year in years:
                    regional_averages_list = self.data["data"][region]["results"][year][month]
                    y.append(sum(regional_averages_list)/len(regional_averages_list))

                y = np.array(y)

                result = mk.original_test(y)
                all_data.append([region, month, result.trend, result.h, result.p, result.slope])

        df = pd.DataFrame(all_data, columns = ["region", "month", "trend", "h", "p", "slope"])

        # create plotly choropleth map
        fig = px.choropleth(
            df,
            locations="region",
            locationmode="USA-states",
            color="slope",
            color_continuous_scale=["green", "yellow", "red"],
            scope="usa",
            title="Mann-Kendall Test Results by State",
            facet_col = "month",
            facet_col_wrap = 4,
            facet_col_spacing = 0
        )

        # Update layout for visual clarity
        fig.update_layout(
            geo=dict(
                lakecolor='rgb(255, 255, 255)',
                bgcolor='rgba(0,0,0,0)'
            )
        )

        fig.show()

class KMeansClustering(Vulcanalyzation):
    def __init__(self, path_to_json = r"MERRA2/JSON Files/Regional Aggregates/us-states-regions.json", regression = "center-and-scale"):
        super().__init__(path_to_json, regression)
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def cluster(self):
        df = self.build_df(self.regression)
        df["cluster"] = None
        for month in self.months:
            month_df = df[df["month"] == month].reset_index(drop = True)

            bottom_tertile = np.percentile(month_df["slope"].values, 33)
            middle_tertile = np.percentile(month_df["slope"].values, 66)
            upper_tertile = np.percentile(month_df["slope"].values, 100)

            month_df["cluster"] = month_df["slope"].map(lambda x: 1 if x < bottom_tertile else 2 if bottom_tertile <= x < middle_tertile else 3)
            print(month_df[["region", "slope", "cluster"]])

            df.loc[df["month"] == month, "cluster"] = month_df["cluster"].values

        cluster_df = df[["slope", "cluster"]].groupby("cluster").median().reset_index()
        fig0 = go.Figure()
        fig0.add_trace(go.Box(x = df["cluster"].values, y = df["slope"].values, name = "Median Slope by Cluster"))
        fig0.show()

        # Create the choropleth map
        fig = px.choropleth(
            df,
            facet_col = "month",
            facet_col_wrap = 4,
            locations='region',  # use state abbreviations for the map
            locationmode='USA-states',  # define the location mode
            color='cluster',  # column to determine the color
            scope='usa',  # focus on the USA
            labels={'cluster': 'Cluster'},  # label for color legend
            title='US States Colored by Cluster'
        )

        # Update layout
        fig.update_layout(
            geo=dict(bgcolor='rgba(0,0,0,0)'),
            title_font_size=20
        )
        fig.show()

class NeuralNetForSlopePrediction(Vulcanalyzation):
    def __init__(self, path_to_json = r"MERRA2/JSON Files/Regional Aggregates/us-states-regions.json", regression = "center-and-scale"):
        super().__init__(path_to_json, regression)

    def train_model(self):
        df = self.build_df(self.regression)
        df["label"] = ((df["p_value"] < 0.05) & (df["slope"] > np.percentile(df["slope"], 70))).astype(int)

        # One-hot encode the 'season' variable
        season_dummies = pd.get_dummies(df['season'], prefix='season')

        # Concatenate the dummy variables with the original DataFrame
        df = pd.concat([df.drop('season', axis=1), season_dummies], axis=1)

        X = df[["num_points", "baseline_temp", "lon_centroid", "lat_centroid", "season_Winter", "season_Spring", "season_Summer", "season_Fall", "cmi", "avg_qv2m"]]
        y = df["slope"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize the model
        model = Sequential()

        # Input layer and first hidden layer
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

        # Second hidden layer
        model.add(Dense(32, activation='relu'))

        # Output layer for regression (no activation function for linear output)
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model (adjust epochs and batch_size as needed)
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        # Evaluate on the test set
        test_loss = model.evaluate(X_test, y_test)

        # Predict on new data
        y_pred = model.predict(X_test)
        y_pred_all = model.predict(X)
        r2 = r2_score(y_test, y_pred)
        print(f'R^2 value for test data: {r2}')

        # plot actual vs predicted
        plt.scatter(y_test, y_pred)
        plt.plot(np.linspace(min(y_test), max(y_test), 100), np.linspace(min(y_test), max(y_test), 100), color = "red")
        plt.xlabel('Actual Slope Values')
        plt.ylabel('Predicted Slope Values')
        plt.title('Actual vs. Predicted Slope')
        plt.show()

class RegressionModelPublicationPlot:
    def __init__(self, path_to_data, from_app = False, scenario = "ct", show_year_graph = False, end_year = 2050, dataset = "MERRA2", var = "T2MMAX"):
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
        self.dataset = dataset
        self.var = var
        self.years = [str(i) for i in range(1980, 2023)]
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
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

    def get_historical_data(self, region, selected_month):
        y = []
        for year in self.years:
            regional_averages_list = self.regional_averages["data"][region]["results"][year][selected_month]
            y.append(sum(regional_averages_list)/len(regional_averages_list))

        y = np.array(y)
        return y

    def average_monthly_max_temp_regression(self, selected_month):
        from sklearn.preprocessing import StandardScaler
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        full_month_names = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "May": "May", "Jun": "June", "Jul": "July", "Aug": "August", "Sep": "September", "Oct": "October", "Nov": "November", "Dec": "December"}
        by_temp_fig = go.Figure()
        if self.show_year_graph:
            by_year_fig = go.Figure()
        colors = ['#00429d', '#3a4198', '#533f94', '#673e90', '#783b8b', '#883887', '#963582', '#a4307e', '#b12a79', '#be2375', '#ca1770', '#d6006c']

        # construct y (local temperatures)
        years = [str(i) for i in range(1980, 2023)]
        for i, month in enumerate(months):
            if month == selected_month:
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

                by_temp_fig.add_trace(go.Scatter(x = X_historical, y = y, name = month, mode = "markers", showlegend = False, marker = dict(color = colors[0])))
                by_temp_fig.add_trace(go.Scatter(x = X_historical, y = y_pred_historical, mode = "lines", showlegend = False, marker = dict(color = colors[0])))
                by_temp_fig.update_xaxes(title = f"p-value = {p_value:.4f}", showline = True, showgrid = False, linecolor = "black", ticks = "outside")

                if self.show_year_graph:
                    by_year_fig.add_trace(go.Scatter(x = years, y = y, name = month, mode = "markers", showlegend = False, marker = dict(color = colors[0])))
                    by_year_fig.update_xaxes(tick0 = 1980, dtick=10, showline = True, showgrid = False)

            by_temp_fig.update_layout(height = 634, width = 600, title = dict(text = f"{full_month_names[selected_month]}", x = 0.5), plot_bgcolor = "white")
            by_temp_fig.update_yaxes(title = "{} Average Max Temperature (°C)".format(full_month_names[selected_month]), showgrid = False, showline = True, linecolor = "black", ticks = "outside")
            by_temp_fig.add_annotation(text = "Global Yearly Average Max Temperature (°C)", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

            by_year_fig.update_layout(height = 634, width = 600, title = dict(text = f"{full_month_names[selected_month]}", x = 0.5), plot_bgcolor = "white")
            by_year_fig.update_yaxes(title = "{} Average Max Temperature (°C)".format(full_month_names[selected_month]), showgrid = False, showline = True, linecolor = "black", ticks = "outside")
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

                by_year_fig.update_xaxes(tick0 = 1980, dtick=10)

        by_temp_fig.update_layout(height = 650, width = 910, title = f"{selected_month}")
        by_temp_fig.update_yaxes(title = "Regional Monthly Average Max Temperature (°C)", row = 2, col = 1)
        by_temp_fig.add_annotation(text = "Global Yearly Average Max Temperature (°C)", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

        if self.show_year_graph:
            by_year_fig.update_layout(height = 650, width = 910, title = "Accelerated Actions Difference From Current Trends Baseline")
            by_year_fig.update_yaxes(title = "Difference in Average Max Daily Temperature (°C)", row = 2, col = 1)
            by_year_fig.add_annotation(text = "Year", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

            return by_temp_fig, by_year_fig
        else:
            return by_temp_fig

    def get_saved_parameters(self):
        return pd.read_csv(r"Regression Results/{}/regression_results-{}-{}.csv".format(self.dataset, self.dataset, self.var))

    def fig_from_saved_parameters(self, selected_region):
        from sklearn.preprocessing import StandardScaler
        from plotly.subplots import make_subplots
        # use this function to quickly generate plot from saved parameters in the relevant CSV file
        # does not generate predictions
        self.parameters = self.get_saved_parameters()

        fig = make_subplots(rows = 3, cols = 4, subplot_titles = self.months, vertical_spacing = 0.1, shared_xaxes = True)
        for i, month in enumerate(self.months):
                
            # get the slope and intercept for the selected month and region
            historical_data = self.get_historical_data(selected_region, month)
            slope = self.parameters[(self.parameters["Month"] == month) & (self.parameters["Region"] == selected_region)]["Slope"].values[0]
            intercept = self.parameters[(self.parameters["Month"] == month) & (self.parameters["Region"] == selected_region)]["Intercept"].values[0]

            # construct historical regression line
            X_historical = self.global_average_temp_by_year_df["Average"].values
            if month == "Jun":
                print(X_historical)
                print(historical_data)
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_historical.reshape(-1, 1))
            y_scaled = scaler_y.fit_transform(historical_data.reshape(-1, 1))
            y_pred_historical = slope*X_scaled + intercept
            y_pred_historical = scaler_y.inverse_transform(y_pred_historical).reshape(-1)

            # make plot
            fig.add_trace(go.Scatter(x = X_historical, y = historical_data, mode = "markers", marker = dict(color = "blue")), row = i//4 + 1, col = i%4 + 1)
            fig.add_trace(go.Scatter(x = X_historical, y = y_pred_historical, mode = "lines", marker = dict(color = "orange")), row = i//4 + 1, col = i%4 + 1)

        fig.add_annotation(text = "Global Yearly Average Max Temperature (°C)", xref = "paper", yref = "paper", x = 0.5, y = -0.1, showarrow = False, font = dict(size = 14))
        fig.update_yaxes(title = "Regional Monthly Average Max Temperature (°C)", showgrid = False, showline = False, linecolor = "black", ticks = "outside", row = 2, col = 1)
        fig.update_layout(title = f"North Dakota Regression Results", showlegend = False, height = 700, width = 1000)
            # fig.update_xaxes(title = f"p-value = {p_value:.4f}", showline = True, showgrid = False, linecolor = "black", ticks = "outside")
            # fig.update_yaxes(title = "{} Average Max Temperature (°C)".format(full_month_names[selected_month]), showgrid = False, showline = True, linecolor = "black", ticks = "outside")
            # fig.add_annotation(text = "Global Yearly Average Max Temperature (°C)", xref = "paper", yref = "paper", x = 0.5, y = -0.15, showarrow = False, font = dict(size = 14))

        return fig

    def main(self):
        if self.scenario == "diff":
            fig1, fig2 = self.average_monthly_max_temp_regression_diff()
        else:
            fig1, fig2 = self.average_monthly_max_temp_regression()

        return fig1, fig2

class RegressionAnalysisComplete:
    def __init__(self, dataset = "MERRA2", var = "T2MMAX", merra_2_timeframe = False):
        self.dataset = dataset
        self.var = var
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if self.dataset == "MERRA2":
            if self.var == "T2MMAX":
                self.data = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/us-states-regions.json"))
            elif self.var == "T2MMEAN":
                self.data = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/us-states-regions-t2m-mean.json"))
            elif self.var == "T2MMIN":
                self.data = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/us-states-regions-t2m-min.json"))
            else:
                raise ValueError
        if self.dataset == "ERA5" and self.var == "T2MMAX":
            self.data = json.load(open(r"ERA5/Temperature Data/JSON Files/us-states-era5-t2m.json"))
        self.merra_2_timeframe = merra_2_timeframe
        self.years = self.data["years"] if not self.merra_2_timeframe else [i for i in range(1980, 2023)]

    def center_and_scale(self, X, y):
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X.reshape(-1, 1))
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        return X_scaled, y_scaled, scaler_X, scaler_y

    def scale_and_center_regression(self, X, y):
        from scipy.stats import linregress
        X_scaled, y_scaled, scaler_X, scaler_y = self.center_and_scale(X, y)

        regression_result = linregress(X_scaled.reshape(-1), y_scaled.reshape(-1))
        slope, intercept, r_value, p_value, std_coef, std_intercept = regression_result.slope, regression_result.intercept, regression_result.rvalue, regression_result.pvalue, regression_result.stderr, regression_result.intercept_stderr

        return slope, intercept, r_value, p_value, std_coef, std_intercept, scaler_X, scaler_y

    def regression_parameters_in_original_units(self, original_X, original_y, scaler_X, scaler_y):
        slope, intercept, r_value, p_value, std_coef, std_intercept, scaler_X, scaler_y = self.scale_and_center_regression(original_X, original_y)

        slope_in_original_units = slope*scaler_y.scale_/scaler_X.scale_
        intercept_in_original_units = scaler_y.mean_ - slope*scaler_X.mean_
        std_coef_in_original_units = std_coef*scaler_y.scale_/scaler_X.scale_
        std_intercept_in_original_units = np.sqrt((scaler_y.scale_*std_intercept)**2 + (scaler_X.mean_*std_coef_in_original_units)**2) # ignoring covariance term

        return slope_in_original_units[0], intercept_in_original_units[0], std_coef_in_original_units[0], std_intercept_in_original_units[0]

    def get_X(self):
        if self.dataset == "MERRA2":
            global_average_temp_by_year_df = pd.read_csv(r"global_average_temp_by_year.csv")
            X = global_average_temp_by_year_df["Average"].values
        elif self.dataset == "ERA5":
            X = []
            json_data = json.load(open(r"ERA5/Temperature Data/JSON Files/world-average.json"))
            for i in json_data:
                if None in i["data"]:
                    i["data"].remove(None)
                year = int(i["name"])
                if year in self.years:
                    X.append(sum(i["data"])/len(i["data"]))
            X = np.array(X)

        return X

    def get_y(self):
        y = []
        years = self.data["years"] if not self.merra_2_timeframe else self.merra_2_timeframe_years
        for year in years:
            y.append(self.data["data"][year]["Average"])
        y = np.array(y)

        return y

    def get_regional_data_from_json(self):
        X = self.get_X()
        all_data = []
        for region in self.data["contains"]:
            # construct y (local temperatures)
            for i, month in enumerate(self.months):
                y = []
                for year in self.years:
                    regional_averages_list = self.data["data"][region]["results"][str(year)][month]
                    y.append(sum(regional_averages_list)/len(regional_averages_list))

                y = np.array(y)

                # this is indeed horribly inefficient, but the runtime is acceptable
                slope, intercept, r_value, p_value, std_coef, std_intercept, scaler_X, scaler_y = self.scale_and_center_regression(X, y)
                slope_in_original_units, intercept_in_original_units, std_coef_in_original_units, std_intercept_in_original_units = self.regression_parameters_in_original_units(X, y, scaler_X, scaler_y)
                all_data.append([region, month, slope, intercept, std_coef, std_intercept, slope_in_original_units, intercept_in_original_units, std_coef_in_original_units, std_intercept_in_original_units, r_value, p_value, scaler_X.mean_[0], scaler_X.scale_[0], scaler_y.mean_[0], scaler_y.scale_[0]])

        df = pd.DataFrame(all_data, columns = ["Region", "Month", "Slope", "Intercept", "Std Coef", "Std Intercept", "Slope in Original Units", "Intercept in Original Units", "Std Coef in Original Units", "Std Intercept in Original Units", "R-Value", "P-Value", "Scaler X Mean", "Scaler X Scale", "Scaler Y Mean", "Scaler Y Scale"])

        return df

    def make_regression_csv(self):
        df = self.get_regional_data_from_json()
        if self.dataset == "MERRA2":
            df.to_csv(f"Regression Results/MERRA2/regression_results-MERRA2-{self.var}.csv", index = False)
        elif self.dataset == "ERA5":
            if self.merra_2_timeframe:
                df.to_csv(f"Regression Results/ERA5/regression_results-ERA5-{self.var}-merra2_timeframe.csv", index = False)
            else:
                df.to_csv(f"Regression Results/ERA5/regression_results-ERA5-{self.var}.csv", index = False)

    def read_regression_csv(self):
        if self.dataset == "MERRA2":
            return pd.read_csv(f"Regression Results/MERRA2/regression_results-MERRA2-{self.var}.csv")
        elif self.dataset == "ERA5":
            if self.merra_2_timeframe:
                return pd.read_csv(r"Regression Results/ERA5/regression_results-ERA5-T2MMAX-merra2_timeframe.csv")
            else:
                return pd.read_csv(r"Regression Results/ERA5/regression_results-ERA5-T2MMAX.csv")

    def make_slope_heatmap(self, export_picture = False, export_svg = False, target = "Slope"):
        slope_df = self.read_regression_csv()

        # create a choropleth map of the trends
        zero_point = abs(min(slope_df[target]))/(max(slope_df[target]) - min(slope_df[target]))
        color_scale = [(0, "#053061"), (zero_point, "white"), (1, "maroon")]
        fig = px.choropleth(
            slope_df,
            locations = "Region",
            locationmode = "USA-states",
            color = target,
            title = f"Trends by Month - {self.dataset}",
            scope = "usa",
            facet_col = "Month",
            facet_col_wrap = 4,
            facet_col_spacing = 0,
            color_continuous_scale = color_scale
        )
        fig.update_layout(width = 1500, height = 1000)
        if export_picture:
            if self.dataset == "MERRA2":
                fig.write_image(f"MERRA2/Figures/slope_magnitude_heatmap-merra2-{target}.png", scale = 2)
            elif self.dataset == "ERA5":
                fig.write_image(f"ERA5/Figures/slope_magnitude_heatmap-era5-{target}.png", scale = 2)
        if export_svg:
            if self.dataset == "MERRA2":
                fig.write_image(f"MERRA2/Figures/slope_magnitude_heatmap-merra2-{target}.svg")
            elif self.dataset == "ERA5":
                fig.write_image(f"ERA5/Figures/slope_magnitude_heatmap-era5-{target}.svg")
        return fig
    
    def yearly_average_regression(self):
        from sklearn.linear_model import LinearRegression
        X = self.get_X()
        all_data = []
        for region in self.data["contains"]:
            # construct y (local temperatures)
            years = self.data["years"] if not self.merra_2_timeframe else self.merra_2_timeframe_years
            state_yearly_averages = []
            for year in years:
                yearly_data = []
                for month in self.months:
                    yearly_data += self.data["data"][region]["results"][str(year)][month]

                state_yearly_averages.append(sum(yearly_data)/len(yearly_data))

            slope = LinearRegression().fit(X.reshape(-1, 1), state_yearly_averages).coef_[0]
            all_data.append([region, slope])

        df = pd.DataFrame(all_data, columns = ["Region", "Slope"])
        return df
    
    def yearly_average_regression_plot(self):
        df = self.yearly_average_regression()
        zero_point = abs(min(df["Slope"]))/(max(df["Slope"]) - min(df["Slope"]))
        color_scale = [(0, "#053061"), (zero_point, "white"), (1, "maroon")]
        fig = px.choropleth(df, 
                            locations = "Region", 
                            locationmode = "USA-states", 
                            color = "Slope", 
                            scope = "usa", 
                            title = "Yearly Average Regression",
                            height = 1000,
                            width = 1500,
                            color_continuous_scale = color_scale
                            )
        return fig

    def mann_kendall_test(self, export_picture = False, export_svg = False):
        import pymannkendall as mk
        all_data = []
        for region in self.data["contains"]:

            # construct y (local temperatures)
            years = [str(i) for i in range(1980, 2023)]
            for i, month in enumerate(self.months):
                y = []
                for year in years:
                    regional_averages_list = self.data["data"][region]["results"][year][month]
                    y.append(sum(regional_averages_list)/len(regional_averages_list))

                y = np.array(y)

                result = mk.original_test(y)
                all_data.append([region, month, result.trend, result.h, result.p, result.slope])

        df = pd.DataFrame(all_data, columns = ["region", "month", "trend", "h", "p", "slope"])

        # create plotly choropleth map
        zero_point = abs(min(df["slope"]))/(max(df["slope"]) - min(df["slope"]))
        color_scale = [(0, "#053061"), (zero_point, "white"), (1, "maroon")]
        fig = px.choropleth(
            df,
            locations="region",
            locationmode="USA-states",
            color="slope",
            color_continuous_scale=color_scale,
            scope="usa",
            title=f"Mann-Kendall Test Results by State - {self.dataset} Data",
            facet_col = "month",
            facet_col_wrap = 4,
            facet_col_spacing = 0
        )

        # Update layout for visual clarity
        fig.update_layout(
            geo=dict(
                lakecolor='rgb(255, 255, 255)',
                bgcolor='rgba(0,0,0,0)'
            ),
            height = 1000,
            width = 1500
        )

        if export_picture:
            if self.dataset == "MERRA2":
                fig.write_image(r"MERRA2/Figures/mann_kendall_test-merra2.png", scale = 2)
            elif self.dataset == "ERA5":
                fig.write_image(r"ERA5/Figures/mann_kendall_test-era5.png", scale = 2)
        if export_svg:
            if self.dataset == "MERRA2":
                fig.write_image(r"MERRA2/Figures/mann_kendall_test-merra2.svg")
            elif self.dataset == "ERA5":
                fig.write_image(r"ERA5/Figures/mann_kendall_test-era5.svg")

        return fig
    
    def order_months(self, data_dict):
        from collections import OrderedDict
        return OrderedDict(sorted(data_dict.items(), key = lambda x: self.months.index(x[0])))

    def mann_kendall_test_on_full_timeseries(self, export_picture = False, export_svg = False):
        import pymannkendall as mk
        years = self.data["years"]
        all_data = []
        for region in self.data["contains"]:
            timeseries = []

            # construct y (local temperatures)
            for year in years:
                ordered_months = self.order_months(self.data["data"][region]["results"][str(year)])
                for month, month_data in ordered_months.items():
                    timeseries.extend(month_data)

            y = np.array(timeseries)

            result = mk.original_test(y)
            all_data.append([region, result.trend, result.h, result.p, result.slope])

        df = pd.DataFrame(all_data, columns = ["region", "trend", "h", "p", "slope"])

        # create plotly choropleth map
        fig = px.choropleth(
            df,
            locations="region",
            locationmode="USA-states",
            color="slope",
            color_continuous_scale=["green", "yellow", "red"],
            scope="usa",
            title=f"Mann-Kendall Test Results on Full Timeseries by State - {self.dataset} Data"
        )

        # Update layout for visual clarity
        fig.update_layout(
            geo=dict(
                lakecolor='rgb(255, 255, 255)',
                bgcolor='rgba(0,0,0,0)'
            )
        )

        if export_picture:
            fig.write_image(r"MERRA2/Figures/mann_kendall_test-merra2.png", scale = 2)
        if export_svg:
            fig.write_image(r"MERRA2/Figures/mann_kendall_test-merra2.svg")

        return fig
    
    def get_y_for_validation(self, region, month):
        y = []
        for year in self.years:
            regional_averages_list = self.data["data"][region]["results"][str(year)][month]
            y.append(sum(regional_averages_list)/len(regional_averages_list))

        y = np.array(y)

        return y

    def validation(self):
        from sklearn.linear_model import LinearRegression
        from scipy.stats import pearsonr

        X = self.get_X()

        all_data = []
        for region in self.data["contains"]:
            for month in self.months:
                y = self.get_y_for_validation(region, month)

                model = LinearRegression().fit(X.reshape(-1, 1), y.reshape(-1, 1))
                y_pred = model.predict(X.reshape(-1, 1))

                r_val = pearsonr(y, y_pred.reshape(-1))
                r_squared = r_val[0]**2
                all_data.append([region, month, r_squared])

        df = pd.DataFrame(all_data, columns = ["Region", "Month", "R^2 Score"])
        return df
    
    def plot_validation(self):
        df = self.validation()
        fig = px.choropleth(df, 
                            locations = "Region", 
                            locationmode = "USA-states", 
                            color = "R^2 Score", 
                            scope = "usa", 
                            title = f"Validation of {self.dataset} Regression Model With R^2 Score",
                            facet_col = "Month",
                            facet_col_wrap = 4,
                            facet_col_spacing = 0
                            )
        fig.update_layout(width = 1200, height = 800)
        return fig

class CompareRegressionResults:
    def __init__(self, merra2_path, era5_path):
        self.merra2_path = merra2_path
        self.era5_path = era5_path
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def read_csvs(self):
        merra2_df = pd.read_csv(self.merra2_path)
        era5_df = pd.read_csv(self.era5_path)

        return merra2_df, era5_df

    def process_slope_df(self):
        merra2_df, era5_df = self.read_csvs()

        merra2_slope_df = merra2_df[merra2_df["Region"].isin(era5_df["Region"])].sort_values(by = ["Region", "Month"]).reset_index(drop = True)
        era5_slope_df = era5_df.sort_values(by = ["Region", "Month"]).reset_index(drop = True)

        return merra2_slope_df, era5_slope_df

    def compare_results_70th_percentile(self):
        merra2_slope_df, era5_slope_df = self.process_slope_df()

        merra2_slope_df["Above 70th Percentile"] = merra2_slope_df["Slope"] > np.percentile(merra2_slope_df["Slope"], 70)
        era5_slope_df["Above 70th Percentile"] = era5_slope_df["Slope"] > np.percentile(era5_slope_df["Slope"], 70)
        merra2_month_proportions = merra2_slope_df.groupby("Month")["Above 70th Percentile"].sum()/(576*0.3)
        era5_month_proportions = era5_slope_df.groupby("Month")["Above 70th Percentile"].sum()/(576*0.3)

        # sort the months by order in the year
        merra2_month_proportions = merra2_month_proportions.reindex(self.months)
        era5_month_proportions = era5_month_proportions.reindex(self.months)

        fig = go.Figure()
        fig.add_trace(go.Bar(x = merra2_month_proportions.index, y = merra2_month_proportions, name = "MERRA2"))
        fig.add_trace(go.Bar(x = era5_month_proportions.index, y = era5_month_proportions, name = "ERA5"))
        fig.update_layout(title = "Distribution of Months with Slopes > 70th Percentile", xaxis_title = "Month", yaxis_title = "Proportion")
        fig.show()

    def compare_results_magnitude_of_slope(self):
        merra2_slope_df, era5_slope_df = self.process_slope_df()

        merra2_slope_df_month_average = merra2_slope_df.groupby("Month")["Slope"].mean()
        era5_slope_df_month_average = era5_slope_df.groupby("Month")["Slope"].mean()

        # sort the months by order in the year
        merra2_slope_df_month_average = merra2_slope_df_month_average.reindex(self.months)
        era5_slope_df_month_average = era5_slope_df_month_average.reindex(self.months)

        fig = go.Figure()
        fig.add_trace(go.Bar(x = merra2_slope_df_month_average.index, y = merra2_slope_df_month_average, name = "MERRA2"))
        fig.add_trace(go.Bar(x = era5_slope_df_month_average.index, y = era5_slope_df_month_average, name = "ERA5"))
        fig.update_layout(title = "Average Slope Magnitude by Month", xaxis_title = "Month", yaxis_title = "Average Slope Magnitude")
        fig.show()

    def compare_slopes_with_sd(self):
        merra2_slope_df, era5_slope_df = self.process_slope_df()
        
        era5_slope_df["SD Distance"] = (era5_slope_df["Slope"] - merra2_slope_df["Slope"])/era5_slope_df["Std Coef"]

        # sort by month order
        era5_slope_df = era5_slope_df.sort_values(by = ["Month"], key = lambda x: x.map(lambda y: self.months.index(y)))

        fig = px.choropleth(
            era5_slope_df,
            locations = "Region",
            locationmode = "USA-states",
            color = "SD Distance",
            scope = "usa",
            title = "SD Distance of Slopes",
            height = 1000,
            width = 1500,
            color_continuous_scale = "RdBu",
            facet_col = "Month",
            facet_col_wrap = 4,
            facet_col_spacing = 0
        )
        
        return fig

class Misc:
    def __init__(self):
        pass

    def compare_average_global_temps(self):
        era5_average_temps = RegressionAnalysisComplete(dataset = "ERA5", var = "T2MMAX").get_X()
        merra2_average_temps = RegressionAnalysisComplete(dataset = "MERRA2", var = "T2MMAX").get_X()

        era5_slice = era5_average_temps[-44:-1]
        merra2_slice = merra2_average_temps[-43:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x = [i for i in range(1980, 2023)], y = era5_slice, name = "ERA5"))
        fig.add_trace(go.Scatter(x = [i for i in range(1980, 2023)], y = merra2_slice, name = "MERRA2"))
        fig.update_layout(title = "Global Average Temperature by Year", xaxis_title = "Year", yaxis_title = "Temperature (°C)")
        fig.show()

    def modified_regression(self):
        regression_df = RegressionAnalysisComplete(dataset = "MERRA2", var = "T2MMAX").get_regional_data_from_json()

        zero_point = abs(min(regression_df["Slope"]))/(max(regression_df["Slope"]) - min(regression_df["Slope"]))
        color_scale = [(0, "#053061"), (zero_point, "white"), (1, "#67001f")]
        fig = px.choropleth(
            regression_df,
            locations = "Region",
            locationmode = "USA-states",
            color = "Slope",
            scope = "usa",
            title = "ERA5 Global Temperature vs. Local MERRA2 Daily Max",
            height = 1000,
            width = 1500,
            color_continuous_scale = color_scale,
            facet_col = "Month",
            facet_col_wrap = 4,
            facet_col_spacing = 0
        )
        
        return fig

class ERA5:
    def __init__(self, raw_max_temp_path = r"ERA5/Temperature Data/Raw Max Temp", raw_average_temp_path = r"ERA5/Temperature Data/Raw Average Temp"):
        self.raw_max_temp_path = raw_max_temp_path
        self.raw_average_temp_path = raw_average_temp_path
        self.years = [i for i in range(1940, 1956)]
        self.data = json.load(open(r"ERA5/Temperature Data/JSON Files/us-states-era5-t2m.json"))
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def read_raw_max_temp(self, file_name):
        year_data = nc.Dataset(os.path.join(self.raw_max_temp_path, file_name))
        return year_data
    
    def read_raw_average_temp(self, file_name):
        year_data = nc.Dataset(os.path.join(self.raw_average_temp_path, file_name))
        return year_data

    def get_data(self, file_path, day):
        dataset = nc.Dataset(file_path)

        return dataset["latitude"][:], dataset["longitude"][:], dataset["t2m"][day, :, :]
    
    def make_dataframe(self, file_path, day):
        lat, lon, temp = self.get_data(file_path, day)
        # Create a DataFrame
        df = pd.DataFrame(columns = lon, index = lat, data = temp)

        return df
    
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

    def global_average_temp(self):
        all_data = {}
        for year in range(1940, 1956):
            if self.is_leap_year(year):
                days = 366
            else:
                days = 365
            average_temps = []
            for day in range(days):
                # compute area weighted average
                df = self.make_dataframe(os.path.join(self.raw_average_temp_path, f"average_{year}.nc"), day)
                area_weighted_average = self.area_weighted_temperatures(df)
                average_temps.append(area_weighted_average)

            all_data[year] = average_temps

        return all_data
    
    def plot_average_temps(self):
        average_temps = self.global_average_temp()
        fig = go.Figure()
        for year, temps in average_temps.items():
            fig.add_trace(go.Scatter(x = [i for i in range(len(temps))], y = temps, name = f"Year {year}"))
        fig.update_layout(title = "Global Average Temperature by Year", xaxis_title = "Day of Year", yaxis_title = "Temperature (°C)")
        fig.show()

    def center_and_scale_regression(self, X, y):
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import linregress

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X.reshape(-1, 1))
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
        regression_result = linregress(X_scaled.reshape(-1), y_scaled.reshape(-1))

        return regression_result.slope, regression_result.intercept, regression_result.rvalue, regression_result.pvalue, regression_result.stderr, regression_result.intercept_stderr
    
    def regression_from_1980(self):
        era_5 = RegressionAnalysisComplete(dataset = "ERA5", var = "T2MMAX")
        X = era_5.get_X()[-44:-1]

        all_data = []
        for region in self.data["contains"]:
            # construct y (local temperatures)
            years = [i for i in range(1980, 2023)]
            for i, month in enumerate(self.months):
                y = []
                for year in years:
                    regional_averages_list = self.data["data"][region]["results"][str(year)][month]
                    y.append(sum(regional_averages_list)/len(regional_averages_list))

                y = np.array(y)

                slope, intercept, r_value, p_value, std_coef, std_intercept = self.center_and_scale_regression(X, y)

                all_data.append([region, month, slope, intercept, r_value, p_value, std_coef, std_intercept])

        df = pd.DataFrame(all_data, columns = ["Region", "Month", "Slope", "Intercept", "R-Value", "P-Value", "Std Coef", "Std Intercept"])
        return df
    
    def plot_regression_results(self):
        df = self.regression_from_1980()
        zero_point = abs(min(df["Slope"]))/(max(df["Slope"]) - min(df["Slope"]))
        color_scale = [(0, "#053061"), (zero_point, "white"), (1, "maroon")]        
        fig = px.choropleth(
            df,
            locations = "Region",
            locationmode = "USA-states",
            color = "Slope",
            scope = "usa",
            title = "ERA5 Global Temperature vs. Local MERRA2 Daily Max",
            facet_col = "Month",
            facet_col_wrap = 4,
            facet_col_spacing = 0,
            color_continuous_scale = color_scale
        )
        
        return fig
    
    def plot_state_data(self):
        df = self.regression_from_1980()
        fig = px.choropleth(
            df,
            locations = "Region",
            locationmode = "USA-states",
            color = "Slope",
            scope = "usa",
            title = "ERA5 Global Temperature vs. Local MERRA2 Daily Max",
            facet_col = "Month",
            facet_col_wrap = 4,
            facet_col_spacing = 0
        )

    def double_check_regression_results(self):
        from sklearn.linear_model import LinearRegression
        # X data
        global_average_temps = json.load(open(r"ERA5/Temperature Data/JSON Files/world-average.json"))
        X = []
        for entry in global_average_temps:
            year = int(entry["name"])
            if 1980 <= year <= 2022:
                year_data = [i for i in entry["data"] if i is not None]
                average = sum(year_data)/len(year_data)
                X.append(average)
        X = np.array(X)

        # y data
        state_data = json.load(open(r"ERA5/Temperature Data/JSON Files/us-states-era5-t2m.json"))
        all_data = []
        for region in state_data["contains"]:
            for month in self.months:
                y = []
                for year in range(1980, 2023):
                    data = state_data["data"][region]["results"][str(year)][month]
                    average = sum(data)/len(data)
                    y.append(average)

                y = np.array(y)
                regression = LinearRegression().fit(X.reshape(-1, 1), y.reshape(-1, 1))
                all_data.append([region, month, regression.coef_[0][0], regression.intercept_[0]])

        df = pd.DataFrame(all_data, columns = ["Region", "Month", "Slope", "Intercept"])

        # plot
        fig = px.choropleth(
            df,
            locations = "Region",
            locationmode = "USA-states",
            color = "Slope",
            scope = "usa",
            title = "ERA5 Global Temperature vs. Local MERRA2 Daily Max",
            facet_col = "Month",
            facet_col_wrap = 4,
            facet_col_spacing = 0
        )
        return fig

class RiskAssessment:
    def __init__(self, dataset = "MERRA2", var = "T2MMAX", state = "MA"):
        self.dataset = dataset
        self.var = var
        self.state = state
        self.abbreviation_dict = self.abbreviation_to_full_name()
        self.state_populations = pd.read_csv(r"state_populations.csv")
        self.state_flowers = pd.read_csv(r"state_flowers.csv")
        self.full_state_name = self.abbreviation_dict[self.state]
    
    def get_regression_results(self):
        if self.dataset == "MERRA2":
            if self.var == "T2MMAX":
                data = pd.read_csv(r"Regression Results/MERRA2/regression_results-MERRA2-T2MMAX.csv")
            elif self.var == "T2MMIN":
                data = pd.read_csv(r"Regression Results/MERRA2/regression_results-MERRA2-T2MMIN.csv")
            elif self.var == "T2MMEAN":
                data = pd.read_csv(r"Regression Results/MERRA2/regression_results-MERRA2-T2MMEAN.csv")
        elif self.dataset == "ERA5":
            if self.var == "T2MMAX":
                data = pd.read_csv(r"Regression Results/ERA5/Max Temp/regression_results-era5.csv")
            elif self.var == "T2MMIN":
                data = pd.read_csv(r"Regression Results/ERA5/Min Temp/regression_results-era5.csv")
            elif self.var == "T2MMEAN":
                data = pd.read_csv(r"Regression Results/ERA5/Mean Temp/regression_results-era5.csv")

        return data

    def abbreviation_to_full_name(self):
        df = pd.read_csv(r"state_cmi.csv")
        abbreviation_dict = {key: value for value, key in zip(df["state"], df["abbreviation"])}

        return abbreviation_dict

    def get_risk_assessment(self):
        regression_data = self.get_regression_results()
        lower_third_percentile = np.percentile(regression_data.groupby("Region")["Slope"].mean(), 33.33)
        upper_third_percentile = np.percentile(regression_data.groupby("Region")["Slope"].mean(), 66.66)
        coef = regression_data[regression_data["Region"] == self.state]["Slope"].values.mean()

        if coef > upper_third_percentile:
            return "HIGH", "red"
        elif lower_third_percentile <= coef <= upper_third_percentile:
            return "MEDIUM", "orange"
        else:
            return "LOW", "green"

    def risk_assessment_div_element(self, n_clicks):
        from dash import html
        from dash_mantine_components import Text

        risk, color = self.get_risk_assessment()

        div_element = html.Div(
            children = [Text(f"{self.full_state_name}", className = "animate__animated animate__fadeInRightBig animate__slow", style = {"fontSize": 30, "color": "black"}, id = f"state-name-{n_clicks}"),
                        Text(f"Population: {self.state_populations[self.state_populations['State'] == self.full_state_name]['Population'].values[0]} ｜ State flower: {self.state_flowers[self.state_flowers['State'] == self.full_state_name]['Common name'].values[0]}", className = "animate__animated animate__fadeInRightBig animate__slow", style = {"fontSize": 20, "color": "black"}, id = f"state-info-{n_clicks}"),
                        html.Div(children = [Text(f"Warming Risk:", style = {"fontSize": 20, "color": "black"}, className = "animate__animated animate__fadeInRightBig animate__slow", id = f"risk-label-{n_clicks}"), Text(f"{risk}", className = "animate__animated animate__fadeInRightBig animate__slow", style = {"fontSize": 20, "color": color, "marginLeft": "10px"}, id = f"risk-value-{n_clicks}")], style = {"display": "flex", "alignItems": "left"})]
        )
        return div_element

class PatternFinding:
    def __init__(self, dataset = "MERRA2", var = "T2MMAX", merra_2_timeframe = False):
        self.dataset = dataset
        self.var = var
        self.merra_2_timeframe = merra_2_timeframe
        self.data = self.load_data()
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def load_data(self):
        if self.dataset == "MERRA2":
            df = pd.read_csv(r"Regression Results/MERRA2/regression_results-MERRA2-T2MMAX.csv")
        elif self.dataset == "ERA5":
            if self.merra_2_timeframe:
                df = pd.read_csv(r"Regression Results/ERA5/regression_results-ERA5-T2MMAX-merra2_timeframe.csv")
            else:
                df = pd.read_csv(r"Regression Results/ERA5/regression_results-ERA5-T2MMAX.csv")

        return df
    
    def preprocess_data_for_clustering(self):
        df = self.load_data()

        # order by month
        df = df.sort_values(by = ["Month"], key = lambda x: x.map(lambda y: self.months.index(y)))

        # represent each state by monthly trend vector
        clustering_data = df.pivot(index='Region', columns='Month', values='Slope in Original Units').reindex(columns = self.months)
        clustering_data_state_names = clustering_data.index.values

        return clustering_data.values, clustering_data_state_names
    
    def k_means_clustering(self, n_clusters = 6):
        from sklearn.cluster import KMeans

        clustering_data, clustering_data_state_names = self.preprocess_data_for_clustering()

        kmeans = KMeans(n_clusters = n_clusters, n_init = 10)
        kmeans.fit(clustering_data)
        results_df = pd.DataFrame({"Region": clustering_data_state_names, "Cluster": kmeans.labels_})
        results_df.to_csv(f"clustering_results_{self.dataset}_{self.var}_{kmeans.n_clusters}.csv", index = False)
    
    def plot_clustering_results(self, n_clusters = 6):
        if os.path.exists(f"clustering_results_{self.dataset}_{self.var}_{n_clusters}.csv"):
            results_df = pd.read_csv(f"clustering_results_{self.dataset}_{self.var}_{n_clusters}.csv")
        else:
            self.k_means_clustering(n_clusters = n_clusters)
            results_df = pd.read_csv(f"clustering_results_{self.dataset}_{self.var}_{n_clusters}.csv")
        
        results_df.sort_values(by = ["Cluster"], inplace = True)
        results_df["Cluster"] = results_df["Cluster"] + 1
        results_df["Cluster"] = results_df["Cluster"].astype(str)

        color_map = {"1": "#80ddff", "2": "#bb80ff", "3": "#ffee80", "4": "#4d8599", "5": "#ddff80", "6": "#ffa280"}
        fig = px.choropleth(
            data_frame = results_df,
            locations = "Region",
            locationmode = "USA-states",
            color = "Cluster",
            scope = "usa",
            title = "Clustering Results",
            color_discrete_map = color_map,
            height = 700,
            width = 1000
        )
        fig.update_layout(title = "K-Means Clustering Results")

        return fig

    def plot_highest_lowest_states(self):
        df = self.load_data().groupby("Region")["Slope in Original Units"].mean().reset_index()
        top_5 = df.nlargest(5, 'Slope in Original Units')
        bottom_5 = df.nsmallest(5, 'Slope in Original Units')

        # Combine them into one DataFrame
        result = pd.concat([top_5, bottom_5])
        result["color"] = ["highest" for i in range(len(result))]
        result["color"][5:] = ["lowest" for i in range(5)]
        color_map = {"highest": "maroon", "lowest": "#053061"}
        print(result)

        fig = px.bar(
            data_frame = result,
            x = "Region",
            y = "Slope in Original Units",
            title = "States with Weakest and Strongest Trends",
            color = "color",
            color_discrete_map = color_map
        )
        fig.update_layout(height = 500, width = 750, showlegend = False)

        return fig
    
    def plot_monthly_distributions(self, n_clusters = 6):
        from plotly.subplots import make_subplots
        from plotly.colors import n_colors, hex_to_rgb

        results_df = pd.read_csv(f"clustering_results_{self.dataset}_{self.var}_{n_clusters}.csv")
        results_df["Cluster"] = results_df["Cluster"] + 1
        merged_df = pd.merge(results_df, self.load_data(), on = "Region")
        max_slope = merged_df["Slope in Original Units"].max()
        min_slope = merged_df["Slope in Original Units"].min()
        color_scale = n_colors(hex_to_rgb("#80ddff"), hex_to_rgb("#bb80ff"), 12, colortype = "tuple")
        color_scale = ["rgb" + str(color) for color in color_scale]

        if n_clusters == 6:
            fig = make_subplots(rows = 4, cols = 3, 
                                subplot_titles = [f"{month}" for month in self.months])
        for i, month in enumerate(self.months):
            cluster_distribution = merged_df[merged_df["Month"] == month]
            # cluster_avg = cluster_distribution.groupby(['Month'])['Slope in Original Units'].mean().reset_index()
            fig.add_trace(go.Histogram(x = cluster_distribution["Slope in Original Units"], name = f"{month}", marker_color = color_scale[i]),  row = i // 3 + 1, col = i % 3 + 1)
            fig.update_xaxes(range = [min_slope, max_slope], row = i // 3 + 1, col = i % 3 + 1)
            fig.update_yaxes(range = [0, 20], row = i // 3 + 1, col = i % 3 + 1)
            fig.add_vline(x = cluster_distribution["Slope in Original Units"].median(), row = i // 3 + 1, col = i % 3 + 1, line = dict(color = "orange"))
            fig.add_annotation(x = cluster_distribution["Slope in Original Units"].median() + 0.75, y = 18, text = f"{cluster_distribution['Slope in Original Units'].median():.2f}", 
                               showarrow = False, font = dict(color = "orange"), row = i // 3 + 1, col = i % 3 + 1)
        fig.update_layout(title = f"Slope Distribution by Month", height = 750, width = 850, showlegend = False)

        return fig

    def plot_cluster_members_average_slope(self, n_clusters = 6):
        results_df = pd.read_csv(f"clustering_results_{self.dataset}_{self.var}_{n_clusters}.csv")
        results_df["Cluster"] = results_df["Cluster"] + 1

        merged_df = pd.merge(results_df, self.load_data(), on = "Region")
        fig = go.Figure()
        for cluster in sorted(merged_df["Cluster"].unique()):
            cluster_distribution = merged_df[merged_df["Cluster"] == cluster]
            print(cluster_distribution)
            cluster_avg = cluster_distribution.groupby(['Month'])['Slope in Original Units'].mean().reset_index()
            fig.add_trace(go.Histogram(x = cluster_distribution["Slope in Original Units"], name = f"Cluster {cluster}"))
        cluster_avg = merged_df.groupby(['Cluster'])['Slope in Original Units'].mean().reset_index()
        print(cluster_avg)

        # sort by month
        cluster_avg = cluster_avg.sort_values(by = ["Month"], key = lambda x: x.map(lambda y: self.months.index(y)))

        fig = px.line(cluster_avg, x = 'Month', y = 'Slope in Original Units', color = 'Cluster', title = 'Average Slope by Cluster and Month')
        fig.update_layout(title = "Average Slope by Cluster and Month")

        return fig
    
    def dtw_with_clustering(self):
        from dtw import dtw

        clustering_data, clustering_data_state_names = self.preprocess_data_for_clustering()
        
        for i, state in enumerate(clustering_data_state_names):
            for j, state2 in enumerate(clustering_data_state_names):
                if i != j:
                    d = dtw(clustering_data[i, :], clustering_data[j, :])
                    print(d.index1, d.index2)

class Comparison:
    def __init__(self, var = "T2MMAX", merra_2_timeframe = True):
        self.merra2_data = PatternFinding(dataset = "MERRA2", var = var, merra_2_timeframe = merra_2_timeframe).load_data()
        self.era5_data = PatternFinding(dataset = "ERA5", var = var, merra_2_timeframe = merra_2_timeframe).load_data()
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def compare_slopes(self):
        # sort data frames by region and month
        self.merra2_data = self.merra2_data.sort_values(by = ["Region", "Month"])
        self.era5_data = self.era5_data.sort_values(by = ["Region", "Month"])
        common_regions = set(self.merra2_data["Region"].unique()) & set(self.era5_data["Region"].unique())
        
        self.merra2_data_filtered = self.merra2_data[self.merra2_data["Region"].isin(common_regions)]
        self.era5_data_filtered = self.era5_data[self.era5_data["Region"].isin(common_regions)]
        self.comparison_df = self.merra2_data_filtered[["Region", "Month"]]
        self.comparison_df["Difference"] = self.merra2_data_filtered["Slope in Original Units"].values - self.era5_data_filtered["Slope in Original Units"].values

        # order by month in the year for visual clarity
        self.comparison_df = self.comparison_df.sort_values(by = ["Month"], key = lambda x: x.map(lambda y: self.months.index(y)))

        zero_point = abs(min(self.comparison_df["Difference"]))/(max(self.comparison_df["Difference"]) - min(self.comparison_df["Difference"]))
        color_scale = [(0, "#053061"), (zero_point, "white"), (1, "maroon")]
        fig = px.choropleth(
            data_frame = self.comparison_df,
            locations = "Region",
            locationmode = "USA-states",
            color = "Difference",
            scope = "usa",
            title = "Difference in Slopes between Merra2 and ERA5",
            facet_col = "Month",
            facet_col_wrap = 4,
            facet_col_spacing = 0,
            color_continuous_scale = color_scale
        )
        fig.update_layout(height = 1000, width = 1500)
        return fig

if __name__ == "__main__":
    fig = AppFunctions().make_year_plot("MA", "aa")
    fig.show()