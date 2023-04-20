import numpy as np
import netCDF4 as nC
import datetime


def land_mask(filename='GRACE/LAND_MASK.CRI.nc'):
    """
    according the land nc file distributed by JPL to calculate the land mask
    :param filename: the file name of the nc file
    :return: the land mask  (type: ndarray)
    """
    var_info = nC.Dataset(filename)
    var = 'land_mask'
    mask = var_info[var][:]
    mask = np.array(mask)
    return mask


def scale_factor(filename='GRACE/CLM4.SCALE_FACTOR.JPL.MSCNv02CRI.nc'):
    """
    according the scale factor file distributed by JPL to calculate the scale factor to reduce the leakage error
    :param filename: the file name of the nc file
    :return: the factor (type: ndarray)
    """
    var_info = nC.Dataset(filename)
    var = 'scale_factor'
    factor = var_info[var][:]
    factor = np.array(factor)  # shape: 360*720
    factor[factor <= -99999.0] = 0  # missing flag value=-99999.0
    return factor


def missing_months(filename='GRACE/GRCTellus.JPL.200204_202112.GLO.RL06M.MSCNv02CRI.nc'):
    """
    obtain the missing months of GRACE/GRACE-FO
    :param filename: the file name of the mascon data
    :return: the missing months of GRACE/GRACE-FO in date format (type: list)
    """
    nf = nC.Dataset(filename)
    months_missing = nf.__dict__['months_missing']
    months_missing = months_missing[:-8] + ";" + months_missing[-7:]  # since the data format is not completely regular
    months_missing = months_missing.split(";")
    months_missing = [datetime.datetime.strptime(month, '%Y-%m') for month in months_missing]  # str to datetime
    return months_missing
