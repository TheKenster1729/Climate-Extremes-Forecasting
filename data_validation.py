import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# validation of the .geojson files defining state borders
def plot_state_boundaries_on_us_map(geojson_dir="Geojsons"):
    """
    Plots the boundaries of US states from all GeoJSON files in the given directory onto a USA basemap.
    """
    # Load all state geojsons into a GeoDataFrame list
    state_gdfs = []
    for filename in os.listdir(geojson_dir):
        if filename.endswith(".geojson") or filename.endswith(".json"):
            path = os.path.join(geojson_dir, filename)
            try:
                gdf = gpd.read_file(path)
                # Make sure CRS is WGS84 for all
                gdf = gdf.to_crs(epsg=4326)
                state_gdfs.append(gdf)
            except Exception as e:
                print(f"Failed to read: {filename} ({e})")
    # Combine into one GeoDataFrame
    all_states = gpd.GeoDataFrame(pd.concat(state_gdfs, ignore_index=True), crs="EPSG:4326")

    # Plot US boundaries using Natural Earth "states" as reference
    fig, ax = plt.subplots(figsize=(14, 8))
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    world[world['NAME'] == 'United States of America'].boundary.plot(ax=ax, color='gray', linewidth=1, label = "Independently Defined Borders")

    # Plot the state borders from Geojsons
    all_states.boundary.plot(ax=ax, edgecolor="red", linewidth=1, label = "Dataset Used in Analysis")
    all_states.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1)

    ax.set_xlim([-130, -65])
    ax.set_ylim([23, 50])
    ax.set_title("State Boundaries from GeoJSON Overlayed on US Map")
    ax.legend(title="State Boundaries", loc="lower left")
    ax.axis("off")
    plt.show()

if __name__ == "__main__":
    plot_state_boundaries_on_us_map()