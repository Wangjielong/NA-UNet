import netCDF4 as nC
import numpy as np
from tools import lat_weighted
import calendar
import datetime
import os


def precip2month(boundary_file='yangtze.txt', precip_file='precip/precip.2003.nc', resolution=0.5):
    """
    according the daily precipitation to estimate the monthly precipitation
    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :param precip_file: the file name of the precipitation data
    :param resolution: the spatial resolution of the precipitation data
            default: 0.5 degree
    :return: date (ndarray), the basin-averaged precip (ndarray) and gridded precip (ndarray)
    """
    nf = nC.Dataset(precip_file)
    var1 = 'time'
    var1_data = nf[var1][:]
    time = nC.num2date(var1_data, units='hours since 1900-01-01 00:00:00').data[:, np.newaxis]

    var2 = 'precip'
    var2_data = nf[var2][:]
    # shape: months*lat*lon FillValue is -9.9692100e+36; units: mm
    grid = np.array(var2_data)
    grid = grid / 10  # convert the unit to cm
    # convert FillValue to zero
    grid[grid < -10000] = 0

    grid_weight, total_weight = lat_weighted(boundary_file, resolution)
    grid = grid * grid_weight
    basin = np.sum(grid, axis=(1, 2))[:, np.newaxis] / total_weight

    # obtain the days of every month in specific year
    year = int(precip_file[14:18])
    days_of_month = [calendar.monthrange(year, i)[1] for i in range(1, 13)]
    # obtain the starting and ending index of every month
    index_of_month = [sum(days_of_month[:i + 1]) if i > 0 else days_of_month[0] for i in range(12)]
    index_of_month.insert(0, 0)

    data_len = len(time)
    stop_flag = True
    index = 1
    p_basin = []
    p_grid = []
    date = []

    while stop_flag:
        if data_len < index_of_month[index]:
            print("there is not enough daily data for the month to obtain monthly precipitation")
            stop_flag = False
        else:
            p_basin.append(sum(basin[index_of_month[index - 1]:index_of_month[index]]))
            p_grid.append(sum(grid[index_of_month[index - 1]:index_of_month[index]]))
            date.append(datetime.datetime(year=year, month=index, day=1))
            index += 1
        if index == 13:
            break

    p_basin = np.array(p_basin)
    p_grid = np.array(p_grid)
    date = np.array(date)[:, np.newaxis]

    return date, p_basin, p_grid


def precip2series(boundary_file='yangtze.txt', precip_dir='precip/', resolution=0.5):
    """

    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :param precip_dir: the directory that contains all precipitation data
                default: precip/
    :param resolution: the spatial resolution of the precipitation data
            default: 0.5 degree
    :return: time, p_basin, p_grid (ndarray)
    """
    precip_files = os.listdir(precip_dir)  # a list that contains all precipitation data
    if precip_files[0] == 'data_info.txt':
        del precip_files[0]  # since there is a data information file

    time = []
    p_basin = []
    p_grid = []

    for i in range(len(precip_files)):
        if precip_files[i].endswith('.nc'):
            date, basin, grid = precip2month(boundary_file, precip_dir + precip_files[i], resolution)
            if len(time) == 0:
                time = date
                p_basin = basin
                p_grid = grid
            else:
                time = np.vstack([time, date])
                p_basin = np.vstack([p_basin, basin])
                p_grid = np.vstack([p_grid, grid])
    return time, p_basin, p_grid


# t, pb, pg = precip2series()
