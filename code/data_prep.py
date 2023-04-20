from precip import precip2series
from temperature import temperature2series
from gldas import gldas2series
from twsa import data_frame2array
import numpy as np


# This data preparation aims to crop the image of yangtze for the U-net The time coverage of GRACE is from 2002-04  to
# 2017-06 while GFO is from 2018-06 to 2021-11. Since we used the driving factors (i.e., precipitation, temperature
# and NOAH TWSA ) of last month (i.e., delay=1) to fill the gap, so the time coverage of driving factors is from
# 2002-03 to 2017-05 and from 2018-05 to 2021-10, respectively The original yangtze extent has column index from
# 171-242 and row index from 108 to 130 (index counts from zero) The extent of yangtze river basin is too small,
# so we cropped the yangtze image to a new extent viz., the column index from 163-242 and the row from 103 to 134.
# There the reshaped yangtze has a dimension of 32*80
# GRACE is  2002-04  ~ 2017-06   GFO is  2018-06 ~ 2021-11    data gap is 2017-07 ~ 2018-05
# 1 is for 2002-03 ~ 2017-05     2 is for 2018-05 ~ 2021-10   3 is for 2017-06 ~ 2018-04


def crop_precip(boundary_file='yangtze.txt', precip_dir='precip/', resolution=0.5):
    """
    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :param precip_dir: the directory that contains all precipitation data
                default: precip/
    :param resolution: the spatial resolution of the precipitation data
            default: 0.5 degree
    :return: cropped time, p_basin and p_grid (ndarray)
    """
    time, p_basin, p_grid = precip2series(boundary_file, precip_dir, resolution)
    cropped_time = time[26:263, :]
    cropped_basin = p_basin[26:263, :]
    cropped_grid = p_grid[26:263, 103:135, 163:243]
    return cropped_time, cropped_basin, cropped_grid


def crop_temperature(boundary_file='yangtze.txt',
                     temperature_file='temperature/air.mon.mean.nc', resolution=0.5):
    """
    :param temperature_file: the file name of the temperature data
    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :param resolution: the spatial resolution of temperature data
              default: 0.5 degree
    :return: cropped time, t_basin and t_grid (ndarray)
    """
    time, t_basin, t_grid = temperature2series(boundary_file, temperature_file, resolution)
    cropped_time = time[650:887, :]
    cropped_basin = t_basin[650:887, :]
    cropped_grid = t_grid[650:887, 103:135, 163:243]
    return cropped_time, cropped_basin, cropped_grid


def crop_gldas(boundary_file='yangtze.txt', gldas_dir='GLDAS/', resolution=0.5, para='tws'):
    """
    :param para: to obtain tws or swe or sm or canop
       default: tws
    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :param gldas_dir: the directory that contains gldas data
                default: GLDAS/
    :param resolution: the spatial resolution of the gldas data
            default: 0.5 degree
    :return: cropped time, g_basin and g_grid (ndarray)
    """
    time, g_basin, g_grid = gldas2series(boundary_file, gldas_dir, resolution, para)
    cropped_time = time[26:263, :]
    cropped_basin = g_basin[26:263, :]
    cropped_grid = g_grid[26:263, 103:135, 163:243]
    return cropped_time, cropped_basin, cropped_grid


def crop_grace(boundary_file='yangtze.txt',
               mascon_file='GRACE/GRCTellus.JPL.200204_202112.GLO.RL06M.MSCNv02CRI.nc', resolution=0.5):
    """

    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :param mascon_file: the file name of the mascon data
    :param resolution: the spatial resolution of mascon data
             default: 0.5 degree
    :return: cropped time, basin and grid for grace and grace-fo, respectively
    """
    grace_time, grace_basin, grace_grid, gfo_time, gfo_basin, gfo_grid \
        = data_frame2array(boundary_file, mascon_file, resolution)

    c_grace_time = grace_time
    c_grace_basin = grace_basin
    c_grace_grid = grace_grid[0:183, 103:135, 163:243]

    c_gfo_time = gfo_time[0:42, :]
    c_gfo_basin = gfo_basin[0:42, :]
    c_gfo_grid = gfo_grid[0:42, 103:135, 163:243]
    return c_grace_time, c_grace_basin, c_grace_grid, c_gfo_time, c_gfo_basin, c_gfo_grid


def prep_inputs(boundary_file='yangtze.txt', precip_dir='precip/', temperature_file='temperature/air.mon.mean.nc',
                gldas_dir='GLDAS/', resolution=0.5, para='tws'):
    """
    prepare the inputs for the U-net  training and filling the data gap
    C * H * W = 3 * 32 * 80
    GRACE is  2002-04  ~ 2017-06   GFO is  2018-06 ~ 2021-11    data gap is 2017-07 ~ 2018-05
    1 is for 2002-03 ~ 2017-05     2 is for 2018-05 ~ 2021-10   3 is for 2017-06 ~ 2018-04
    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :param precip_dir: the directory that contains all precipitation data
             default: precip/
    :param temperature_file: the file name of the temperature data
    :param gldas_dir: the directory that contains gldas data
    :param resolution: the spatial resolution of the input
    :param para: to obtain tws or swe or sm or canop from gldas
        default: tws
    :return:
    """
    # precipitation
    p_time, p_basin, p_grid = crop_precip(boundary_file, precip_dir, resolution)
    time1 = p_time[0:183, :]
    time2 = p_time[194:236, :]
    time3 = p_time[183:194, :]
    p_grid1 = p_grid[0:183, :, :]
    p_grid2 = p_grid[194:236, :, :]
    p_grid3 = p_grid[183:194, :, :]

    # temperature
    t_time, t_basin, t_grid = crop_temperature(boundary_file, temperature_file, resolution)
    t_grid1 = t_grid[0:183, :, :]
    t_grid2 = t_grid[194:236, :, :]
    t_grid3 = t_grid[183:194, :, :]

    # GLDAS
    g_time, g_basin, g_grid = crop_gldas(boundary_file, gldas_dir, resolution, para)
    g_grid1 = g_grid[0:183, :, :]
    g_grid2 = g_grid[194:236, :, :]
    g_grid3 = g_grid[183:194, :, :]  # 11*3*32*80   the data used as inputs to fill the data gap

    channel = 3
    sample1 = time1.shape[0]
    sample2 = time2.shape[0]
    sample3 = time3.shape[0]

    grid1 = np.zeros((sample1, channel, 32, 80))
    grid2 = np.zeros((sample2, channel, 32, 80))
    grid3 = np.zeros((sample3, channel, 32, 80))

    for i in range(sample1):
        temp_grid = np.array([p_grid1[i], t_grid1[i], g_grid1[i]])
        grid1[i] = temp_grid

    for i in range(sample2):
        temp_grid = np.array([p_grid2[i], t_grid2[i], g_grid2[i]])
        grid2[i] = temp_grid

    for i in range(sample3):
        temp_grid = np.array([p_grid3[i], t_grid3[i], g_grid3[i]])
        grid3[i] = temp_grid

    return time1, time2, time3, grid1, grid2, grid3


def tr_val_test(inputs='inputs.npz', targets='targets.npz', shuffle=True, data_augmentation=False, sim_flag=False):
    """
    time1, time2, time3, grid1, grid2, grid3 = prep_inputs()
    np.savez('inputs.npz', t1=time1, t2=time2, t3=time3, g1=grid1, g2=grid2, g3=grid3)
    grace_time, grace_basin, grace_grid, gfo_time, gfo_basin, gfo_grid = crop_grace()
    np.savez('targets.npz', grace_t=grace_time, gfo_t=gfo_time, grace=grace_grid, gfo=gfo_grid)
    generate the sets of training(80%), validating(10%), and testing(10%)
    :param sim_flag: simulate the gap
    :param data_augmentation: flip up or left the data
    :param shuffle: to shuffle the data or not
    :param inputs: the .npz file that contains all driving factors data
    :param targets: the .npz file that contains grace and grace_fo data
    :return: trX, trY, valX, valY, testX, testY, inputs_t (training months)
    """
    x = np.load(inputs, allow_pickle=True)
    y = np.load(targets, allow_pickle=True)
    grid1 = x['g1']  # 183*3*32*80   S * C * H * W = S * 3 * 32 * 80  (S for samples)
    grid2 = x['g2']  # 42*3*32*80
    inputs_t1 = x['t1']
    inputs_t2 = x['t2']
    grace_grid = y['grace']  # 183*32*80  S*H*W
    gfo_grid = y['gfo']  # 42*32*80

    if sim_flag:  # simulate the gap from 2016-07~2017-06
        inputs_t1 = inputs_t1[:-12]
        grid1 = grid1[:-12]
        grace_grid = grace_grid[:-12]

    inputs_t = np.concatenate((inputs_t1, inputs_t2), axis=0)  # outputs_t =inputs_t + 1 month
    inputs_grid = np.concatenate((grid1, grid2), axis=0)  # 225*3*32*80
    targets_grid = np.concatenate((grace_grid, gfo_grid), axis=0)  # 225*32*80
    targets_grid = targets_grid[:, np.newaxis, :, :]  # add the channel  225*1*32*80

    if data_augmentation:
        inputs_grid_up = np.flip(inputs_grid, 2)  # flip up
        inputs_grid_left = np.flip(inputs_grid, 3)  # flip left
        targets_grid_up = np.flip(targets_grid, 2)  # flip up
        targets_grid_left = np.flip(targets_grid, 3)  # flip left
        # inputs_grid = np.concatenate((inputs_grid, inputs_grid_up, inputs_grid_left), axis=0)
        # targets_grid = np.concatenate((targets_grid, targets_grid_up, targets_grid_left), axis=0)
        grid_normal = np.random.normal(loc=0, scale=0.5, size=(32, 80))
        inputs_grid_normal = inputs_grid + grid_normal
        targets_grid_normal = targets_grid
        inputs_grid = np.concatenate((inputs_grid, inputs_grid_normal), axis=0)
        targets_grid = np.concatenate((targets_grid, targets_grid_normal), axis=0)
        inputs_t = np.concatenate((inputs_t, inputs_t), axis=0)

    data_len = inputs_grid.shape[0]
    indices = range(data_len)
    if shuffle:
        # set a seed for reproducibility. Since you are generating random values, setting a seed
        # will ensure that the values generated are the same if the seed set is the same each time the code is run
        np.random.seed(1)
        indices = np.random.permutation(range(data_len))

    if sim_flag:
        split_idx1 = int(len(indices) * 0.8)  # training(80%), validating(20%), and testing(0%)
        split_idx2 = int(len(indices) * 1)
    else:
        split_idx1 = int(len(indices) * 0.8)  # training(80%), validating(10%), and testing(10%)
        split_idx2 = int(len(indices) * 0.9)

    trX = inputs_grid[indices[:split_idx1]]
    trY = targets_grid[indices[:split_idx1]]
    input_trT = inputs_t[indices[:split_idx1]]

    valX = inputs_grid[indices[split_idx1:split_idx2]]
    valY = targets_grid[indices[split_idx1:split_idx2]]
    input_valT = inputs_t[indices[split_idx1:split_idx2]]

    testX = inputs_grid[indices[split_idx2:]]
    testY = targets_grid[indices[split_idx2:]]
    input_testT = inputs_t[indices[split_idx2:]]

    return trX, trY, valX, valY, testX, testY, input_trT, input_valT, input_testT


if __name__ == '__main__':
    print('start...')
    tr_val_test(sim_flag=True)
