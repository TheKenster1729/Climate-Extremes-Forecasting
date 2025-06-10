import json

era5_world_average = json.load(open(r"ERA5/Temperature Data/JSON Files/world-average.json", "r"))

temps = []
for year_data in era5_world_average:
    year = year_data["name"]
    data = year_data["data"]
    
    if 1979 < int(year) < 2023:
        temps += data

temps = [i for i in temps if i is not None]

print(sum(temps) / len(temps))