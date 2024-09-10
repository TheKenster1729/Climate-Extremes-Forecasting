import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import netCDF4 as nc
from scipy.stats.qmc import Halton


def load_file(file_path):
    dataset = nc.Dataset(file_path)
    return dataset["T2MMEAN"][:][0]

data = load_file("MERRA2/Temperature Data/Max Temp/1980/MERRA2_100.statD_2d_slv_Nx.19800101.SUB.nc")
lat = np.linspace(0, 360, 300, dtype = int)
lon = np.linspace(0, 575, 300, dtype = int)
grid = np.stack([lat, lon], axis = -1)

data_grid = data[lat[:, np.newaxis], lon]

# interpolation
interpolator = RBFInterpolator(grid, data_grid)
newgrid = np.stack([np.linspace(0, 1000, 100), np.linspace(0, 1000, 100)], axis = -1)
newdata = interpolator(newgrid)

# actual data
fig1, ax1 = plt.subplots()
ax1.pcolormesh(data_grid)
ax1.set_aspect("equal")

# interpolated data
fig2, ax2 = plt.subplots()
ax2.pcolormesh(newdata)
plt.show()
