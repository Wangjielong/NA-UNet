import netCDF4 as nC
import numpy as np
from tools import lat_weighted
from torch.nn.functional import avg_pool2d
import torch
import datetime
import os


def format_change(old_array):
    """
    change the original GLDAS data to world-complete data format
    :param old_array: 1*600*1440
    :return: new_array 1*360*720
    """
    # old_array 1*600*1440 (ndarray)
    filled_value = np.zeros((1, 120, 1440))
    new_array = np.concatenate((filled_value, old_array), axis=1)  # 1*720*1440
    right = new_array[0:1, :, 720:]
    left = new_array[0:1, :, :720]
    new_array = np.concatenate((right, left), axis=2)  # exchange the right with the left
    new_array = np.flip(new_array, axis=1)  # 1*720*1440 # flip up and down
    new_array = torch.from_numpy(new_array.copy())
    # convert the resolution of 0.25deg to 0.5deg
    new_array = avg_pool2d(new_array, kernel_size=2, stride=2)
    new_array = new_array.numpy()  # 1*360*720 (ndarray)
    # missing value=-9999.0
    new_array[new_array <= -9998] = 0
    return new_array


def gldas2month(gldas_file='GLDAS/GLDAS_NOAH025_M.A200301.021.nc4', para='tws'):
    """
    grid(TWS)=
    SoilMoi0_10cm_inst+SoilMoi10_40cm_inst+SoilMoi40_100cm_inst+SoilMoi100_200cm_inst+CanopInt_inst+SWE_inst
    :param para: to obtain tws or swe or sm or canop
        default: tws
    :param gldas_file: the file name of the GLDAS data
    :return: date (type:list), g_grid (type: ndarray 1*360*720)
    """

    pho_water = 1000  # %the density of water  kg/m3
    nf = nC.Dataset(gldas_file)
    data = np.empty((1, 360, 720))
    canop = np.empty((1, 360, 720))
    swe = np.empty((1, 360, 720))
    sm = np.empty((1, 360, 720))
    if para == 'canop' or para == 'tws':
        var = 'CanopInt_inst'
        var_data = np.array(nf[var][:])
        data = format_change(var_data)  # units  kg/m2
        data = data / pho_water * 100  # units  (EWH) cm
        if para == 'tws':
            canop = data
    if para == 'swe' or para == 'tws':
        var = 'SWE_inst'
        var_data = np.array(nf[var][:])
        data = format_change(var_data)  # units  kg/m2
        data = data / pho_water * 100  # units  (EWH) cm
        if para == 'tws':
            swe = data
    if para == 'sm' or para == 'tws':
        var1 = 'SoilMoi0_10cm_inst'
        var2 = 'SoilMoi10_40cm_inst'
        var3 = 'SoilMoi40_100cm_inst'
        var4 = 'SoilMoi100_200cm_inst'

        var1_data = np.array(nf[var1][:])
        var2_data = np.array(nf[var2][:])
        var3_data = np.array(nf[var3][:])
        var4_data = np.array(nf[var4][:])

        data1 = format_change(var1_data)
        data2 = format_change(var2_data)
        data3 = format_change(var3_data)
        data4 = format_change(var4_data)
        data = data1 + data2 + data3 + data4
        data = data / pho_water * 100  # units  (EWH) cm
        if para == 'tws':
            sm = data

    if para == 'tws':
        data = canop + swe + sm

    g_grid = data
    year = int(gldas_file[23:27])
    month = int(gldas_file[27:29])
    date = [datetime.datetime(year=year, month=month, day=1)]

    return date, g_grid


def gldas2series(boundary_file='yangtze.txt', gldas_dir='GLDAS/', resolution=0.5, para='tws'):
    """
    baseline: 2004-01~2009-12
    :param para: to obtain tws or swe or sm or canop
       default: tws
    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :param gldas_dir: the directory that contains gldas data
                default: GLDAS/
    :param resolution: the spatial resolution of the gldas data
            default: 0.5 degree
    :return: time, g_basin, g_grid (type: ndarray)
    """

    gldas_files = os.listdir(gldas_dir)  # a list that contains baseline data
    baseline_grid = np.empty((1, 360, 720))
    num_months = 0
    time = []
    g_basin = []
    g_grid = []

    grid_weight, total_weight = lat_weighted(boundary_file, resolution)

    for i in range(len(gldas_files)):
        if gldas_files[i].endswith('.nc4'):
            date, grid = gldas2month(gldas_dir + gldas_files[i], para)
            if 2004 <= int(gldas_files[i][17:21]) <= 2009:
                baseline_grid += grid
                num_months += 1
            if len(time) == 0:
                time = date
                g_grid = grid
            else:
                time = np.vstack([time, date])
                g_grid = np.concatenate((g_grid, grid), axis=0)

    if num_months != 72:
        print('there is not enough data to estimate the baseline of gldas')
    else:
        baseline_grid = baseline_grid / num_months
        g_grid = g_grid - baseline_grid
        g_grid = g_grid * grid_weight
        g_basin = np.sum(g_grid, axis=(1, 2))[:, np.newaxis] / total_weight

    return time, g_basin, g_grid
