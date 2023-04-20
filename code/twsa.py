from mascon import scale_factor, land_mask, missing_months
from tools import lat_weighted
import netCDF4 as nC
import numpy as np
import pandas as pd
import datetime


def mascon2series(boundary_file='yangtze.txt',
                  mascon_file='GRACE/GRCTellus.JPL.200204_202112.GLO.RL06M.MSCNv02CRI.nc', resolution=0.5,
                  land_file='GRACE/LAND_MASK.CRI.nc', scale_file='GRACE/CLM4.SCALE_FACTOR.JPL.MSCNv02CRI.nc'):
    """
    calculate the TWSA of specified region derived from GRACE and GRACE-Fo with the missing months and gap
    :param mascon_file: the file name of the mascon data
    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :param resolution: the spatial resolution of mascon data
              default: 0.5 degree
    :return: time (ndarray), the basin-averaged twsa (ndarray) and gridded twsa (ndarray)
    """
    # obtain the weights
    grid_weight, total_weight = lat_weighted(boundary_file, resolution)  # shape 360*720
    # obtain the land mask
    lm = land_mask(filename=land_file)  # shape 360*720
    # obtain the scale factor
    sf = scale_factor(filename=scale_file)  # shape 360*720

    nf = nC.Dataset(mascon_file)
    var1 = 'time'
    var1_data = nf[var1][:]
    time = nC.num2date(var1_data, units='days since 2002-01-01 00:00:00').data[:, np.newaxis]

    var2 = 'lwe_thickness'
    var2_data = nf[var2][:]
    twsa_grid = np.array(var2_data)  # shape: months*lat*lon  unit: (EWH)cm
    twsa_grid = twsa_grid * lm * sf
    twsa_grid = np.flip(twsa_grid, 1)  # flip up

    twsa_grid = twsa_grid * grid_weight
    twsa_basin = np.sum(twsa_grid, axis=(1, 2))[:, np.newaxis] / total_weight

    return time, twsa_basin, twsa_grid


def filling_twsa(boundary_file='yangtze.txt',
                 mascon_file='GRACE/GRCTellus.JPL.200204_202112.GLO.RL06M.MSCNv02CRI.nc', resolution=0.5):
    """
    using the linear interpolation method to fill the missing month values of GRACE and GRACE-FO
    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :param mascon_file: the file name of the mascon data
    :param resolution: the spatial resolution of mascon data
             default: 0.5 degree
    :return: GRACE and GRACE-FO data with the missing month(s) filled by the linear interpolation (type: DataFrame)
    """
    time, twsa_basin, twsa_grid = mascon2series(boundary_file, mascon_file, resolution)
    months_missing = missing_months()
    data = pd.DataFrame(columns=['date', 'value', 'grid'])
    for i in range(time.shape[0]):
        temp_date = time[i][0]
        time[i][0] = datetime.datetime(temp_date.year, temp_date.month, 1)
        data = data.append({'date': time[i][0], 'value': twsa_basin[i][0], 'grid': twsa_grid[i]},
                           ignore_index=True)

    helper = pd.DataFrame({'date': months_missing})
    data = pd.merge(data, helper, on='date', how='outer').sort_values('date')
    data = data.reset_index(drop=True)  # we rearrange the index
    grace = data[0:183]
    grace_fo = data[194:].reset_index(drop=True)
    grace = linear_interpolation(grace)
    grace_fo = linear_interpolation(grace_fo)

    return grace, grace_fo


def linear_interpolation(data):
    """
    using the linear interpolation method to fill the missing month values
    the method is applied to the data where the missing month(s) is not more than two months
    using the linear interpolation
    nan represents not a number, so nan != nan
    :param data: the DataFrame with column names having value, and grid (type: DataFrame)
    :return: the complete data with missing months filled by the linear interpolation (type: DataFrame)
    """
    for index, row in data.iterrows():
        if row['value'] != row['value']:
            up_value = data.loc[index - 1]['value']  # the month before the missing month
            up_flag = (up_value == up_value)
            down_value = data.loc[index + 1]['value']  # the month after the missing month
            down_flag = (down_value == down_value)
            if up_flag and down_flag:
                data.at[index, 'value'] = (up_value + down_value) / 2
                data.at[index, 'grid'] = (data.at[index - 1, 'grid'] + data.at[index + 1, 'grid']) / 2

            if up_flag and not down_flag:
                down2_value = data.loc[index + 2]['value']
                down2_flag = (down2_value == down2_value)  # the second month after the missing month
                if up_flag and down2_flag:
                    difference = (down2_value - up_value) / 3
                    data.at[index, 'value'] = up_value + difference
                    data.at[index + 1, 'value'] = down2_value - difference

                    grid_difference = (data.at[index + 2, 'grid'] - data.at[index - 1, 'grid']) / 3
                    data.at[index, 'grid'] = data.at[index - 1, 'grid'] + grid_difference
                    data.at[index + 1, 'grid'] = data.at[index + 2, 'grid'] - grid_difference

    return data


def data_frame2array(boundary_file='yangtze.txt',
                     mascon_file='GRACE/GRCTellus.JPL.200204_202112.GLO.RL06M.MSCNv02CRI.nc', resolution=0.5):
    """
    according to the results from filling_twsa, we convert the data in DataFrame format to ndarray
    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :param mascon_file: the file name of the mascon data
    :param resolution: the spatial resolution of mascon data
             default: 0.5 degree
    :return: grace_time, grace_basin, grace_grid, gfo_time, gfo_basin, gfo_grid (ndarray)
    """
    # DataFrame format
    grace, grace_fo = filling_twsa(boundary_file, mascon_file, resolution)
    grace = grace.values
    grace_fo = grace_fo.values

    data_len = len(grace) if len(grace) > len(grace_fo) else len(grace_fo)
    g_time = grace[0:183, 0][:, np.newaxis]
    g_basin = grace[0:183, 1][:, np.newaxis]
    g_grid = grace[0:183, 2][:, np.newaxis]
    gfo_time = grace_fo[0:43, 0][:, np.newaxis]
    gfo_basin = grace_fo[0:43, 1][:, np.newaxis]
    gfo_grid = grace_fo[0:43, 2][:, np.newaxis]

    grace_time = []
    grace_basin = []
    grace_grid = []
    grace_fo_time = []
    grace_fo_basin = []
    grace_fo_grid = []

    for i in range(data_len):
        time1 = g_time[i][0].to_pydatetime()
        basin1 = g_basin[i][0]
        grid1 = g_grid[i][0][np.newaxis, :, :]
        time2 = []
        basin2 = []
        grid2 = []
        if i < len(grace_fo):
            time2 = gfo_time[i][0].to_pydatetime()
            basin2 = gfo_basin[i][0]
            grid2 = gfo_grid[i][0][np.newaxis, :, :]
        if i == 0:
            grace_time = time1
            grace_basin = basin1
            grace_grid = grid1
            grace_fo_time = time2
            grace_fo_basin = basin2
            grace_fo_grid = grid2
        else:
            grace_time = np.vstack([grace_time, time1])
            grace_basin = np.vstack([grace_basin, basin1])
            grace_grid = np.concatenate((grace_grid, grid1), axis=0)
            if i < len(grace_fo):
                grace_fo_time = np.vstack([grace_fo_time, time2])
                grace_fo_basin = np.vstack([grace_fo_basin, basin2])
                grace_fo_grid = np.concatenate((grace_fo_grid, grid2), axis=0)
    return grace_time, grace_basin, grace_grid, grace_fo_time, grace_fo_basin, grace_fo_grid
