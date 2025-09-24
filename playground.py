import pandas as pd

state = "MO"
month = "Jan"
dataset = "era5"

max_temps = pd.read_csv(r"full_processed_data_t2mmax.csv")
mean_temps = pd.read_csv(r"full_processed_data_t2mmean.csv")
min_temps = pd.read_csv(r"full_processed_data_t2mmin.csv")

max_temp = max_temps[(max_temps["Region"] == state)
                     & (max_temps["Month"] == month)
                     & (max_temps["Dataset"] == dataset)]

mean_temp = mean_temps[(mean_temps["Region"] == state)
                     & (mean_temps["Month"] == month)
                     & (mean_temps["Dataset"] == dataset)]

min_temp = min_temps[(min_temps["Region"] == state)
                     & (min_temps["Month"] == month)
                     & (min_temps["Dataset"] == dataset)]



print(max_temp)
print(mean_temp)
print(min_temp)