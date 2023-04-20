import netCDF4 as nC
import numpy as np
from tools import lat_weighted


def temperature2series(boundary_file='yangtze.txt',
                       temperature_file='temperature/air.mon.mean.nc', resolution=0.5):
    """
    calculate the TWSA of specified region derived from GRACE and GRACE-Fo with the missing months and gap
    :param temperature_file: the file name of the temperature data
    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :param resolution: the spatial resolution of temperature data
              default: 0.5 degree
    :return: time, t_basin, t_grid (ndarray)
    """
    nf = nC.Dataset(temperature_file)

    var1 = 'time'
    var1_data = nf[var1][:]
    time = nC.num2date(var1_data, units='hours since 1800-01-01 00:00:00').data[:, np.newaxis]

    grid_weight, total_weight = lat_weighted(boundary_file, resolution)
    var2 = 'air'
    var2_data = nf[var2][:]
    # shape: months*lat*lon  unit: K; FillValue is -9.9692100e+36;  {[150,400],'valid_range'}
    t_grid = np.array(var2_data)
    t_grid = t_grid - 273.15  # change units of K to ℃
    # convert FillValue to zero
    t_grid[t_grid < -10000] = 0

    t_grid = t_grid * grid_weight
    t_basin = np.sum(t_grid, axis=(1, 2))[:, np.newaxis] / total_weight

    return time, t_basin, t_grid
