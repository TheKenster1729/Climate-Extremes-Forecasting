# Climate-Extremes-Forecasting

## Before Getting Started
Make sure all required packages are installed. Run `pip install -r requirements.txt` to install all required packages.

## Running the Webapp Locally
Run `python app.py` to run the webapp locally (should reflect the live version at extremed.mit.edu).

## Data Location
The data used to create the regressions is located in the Data folder. Both the MERRA2 and ERA5 JSONs, which store local data, follow the same format; global data is found in the CSV files named global_temp.csv in each folder.
(Note that "rescaled" for the ERA5 dataset just means it was resampled to the same resolution as the MERRA2 dataset). The JSON files contain day-by-day data for each region; to access monthly averages (which were used for the regressions),
use the files full_processed_data_t2m{min, mean, max}.csv.

## Additional Tools
The file data_visualization.py contains simple functions to visualize the global temperature and local temperature data. The global temperature function allows you to compare global average temperature data for each dataset (ERA5 and MERRA2). The local temperature function allows you to compare local average temperature data for each dataset (ERA5 and MERRA2) and each temperature variable (minimum, mean, and maximum) for a given region.