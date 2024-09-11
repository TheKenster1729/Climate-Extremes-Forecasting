import geopandas as gpd
import matplotlib.pyplot as plt

# Load the GeoJSON for the U.S. map and the region
us_map = gpd.read_file('assets/gz_2010_us_outline_5m.json')
region = gpd.read_file('Regional Geojsons/lowermississippiregion.geojson')

# Plot the U.S. map
ax = us_map.plot(color='lightgray')

# Highlight the region
region.plot(ax=ax, color='red')

# Save as SVG
plt.savefig('us_map_with_region.svg', format='svg')
