import numpy as np
import torch
from tools import region_mask, lat_weighted, lat_lon_gen
import scipy.io as io
from networks import U_Net

mask = region_mask(boundary_file='yangtze.txt')
mask = mask[103:135, 163:243]


# evaluate the model performance

def corr(targets, predicted):
    """
    estimate the correlation coefficient between the targets and predicted
    :param targets: samples * H * W  (3D ndarray)
    :param predicted: samples * H * W (3D ndarray)
    :return: grid correlation
    """
    h = targets.shape[1]
    w = targets.shape[2]
    corr_grid = np.empty((h, w))

    for i in range(h):
        for j in range(w):
            if mask[i][j] != 0:
                corr_grid[i][j] = np.corrcoef(targets[:, i, j], predicted[:, i, j], rowvar=False)[0][1]

    return corr_grid * mask


def nse(targets, predicted):
    """
    estimate the NSE between the targets and predicted
    :param targets: samples * H * W  (3D ndarray)
    :param predicted: samples * H * W (3D ndarray)
    :return: grid NSE
    """
    h = targets.shape[1]
    w = targets.shape[2]
    nse_grid = np.empty((h, w))
    for i in range(h):
        for j in range(w):
            if mask[i][j] != 0:
                y = targets[:, i, j]
                y_hat = predicted[:, i, j]
                y_avg = np.mean(y)
                delta1 = sum((y - y_hat) * (y - y_hat)) + 0.0001  # avoid the warning that is divide by zero
                delta2 = sum((y - y_avg) * (y - y_avg)) + 0.00001
                nse_grid[i][j] = 1 - delta1 / delta2

    return nse_grid * mask


def nrmse(targets, predicted):
    """
    estimate the nrmse between the targets and predicted
    :param targets: samples * H * W  (3D ndarray)
    :param predicted: samples * H * W (3D ndarray)
    :return: grid nrmse
    """
    h = targets.shape[1]
    w = targets.shape[2]
    s = targets.shape[0]
    nrmse_grid = np.empty((h, w))
    for i in range(h):
        for j in range(w):
            if mask[i][j] != 0:
                y = targets[:, i, j]
                y_hat = predicted[:, i, j]
                nrmse_grid[i][j] = np.sqrt(sum((y - y_hat) * (y - y_hat)) / s) / (np.max(y) - np.min(y) + 0.0001)
    return nrmse_grid * mask


def model_perf(model, model_inputs, targets, boundary_file='yangtze.txt', resolution=0.5, device='cpu'):
    """
    estimate the performance measures between model outputs
    and the targets during training, validating and testing period, respectively
    :param resolution: the grid resolution
    :param device: the default device: cpu
    :param boundary_file: the file name of the interested region
    :param model: the trained model (e.g., U-net )
    :param model_inputs: the model inputs (tensor): S*C*H*W (e.g., tr_x, val_x or test_x)
    :param targets: the real observations (numpy ndarray): S*1*H*W (the default channel =1 )
    :return:corr_grid, nse_grid, nrmse_grid, the time series of model outputs and the targets
    """
    model.eval()  # prep model for evaluation if we have BN and dropout operation
    model.cpu()
    model_outputs = model(model_inputs)
    if device == 'cpu':
        model_outputs = model_outputs.detach().numpy()  # convert the pytorch tensor to numpy ndarray
    else:
        model_outputs = model_outputs.cpu().detach().numpy()

    targets = targets.squeeze()  # remove the channel dimension
    model_outputs = model_outputs.squeeze()  # remove the channel dimension

    corr_grid = corr(targets, model_outputs)
    nse_grid = nse(targets, model_outputs)
    nrmse_grid = nrmse(targets, model_outputs)

    grid_weight, total_weight = lat_weighted(boundary_file, resolution)
    grid_weight = grid_weight[103:135, 163:243]  # crop the grid weight

    targets_grid = targets * grid_weight
    targets_basin = np.sum(targets_grid, axis=(1, 2))[:, np.newaxis] / total_weight  # un-ordered

    model_grid = model_outputs * grid_weight
    model_basin = np.sum(model_grid, axis=(1, 2))[:, np.newaxis] / total_weight  # un-ordered

    return corr_grid, nse_grid, nrmse_grid, model_basin, targets_basin


def fill_gap(model_pt, model_inputs='inputs.npz', boundary_file='yangtze.txt', resolution=0.5, device='cpu',
             dtype=torch.float32):
    """
    using the driving inputs to fill the data gap
    :param dtype: data type
    :param model_pt: the trained model (e.g., U-net )
    :param model_inputs: the model inputs (tensor): S*C*H*W (e.g., tr_x, val_x or test_x)
    :param boundary_file: the file name of the interested region
    :param resolution: the grid resolution
    :param device: the default device: cpu
    :return:
    """
    x = np.load(model_inputs, allow_pickle=True)
    time = x['t3']
    grid3 = x['g3']  # S*C*H*W
    grid3 = torch.from_numpy(grid3).to(device=device, dtype=dtype)  # convert to tensor
    model = U_Net()
    model.load_state_dict(torch.load(model_pt))
    model.eval()
    model_outputs = model(grid3)
    if device == 'cpu':
        model_outputs = model_outputs.detach().numpy()  # convert the pytorch tensor to numpy ndarray
    else:
        model_outputs = model_outputs.cpu().detach().numpy()
    model_outputs = model_outputs.squeeze()  # remove the channel dimension

    grid_weight, total_weight = lat_weighted(boundary_file, resolution)
    grid_weight = grid_weight[103:135, 163:243]  # crop the grid weight
    gap_grid = model_outputs * grid_weight
    gap_basin = np.sum(gap_grid, axis=(1, 2))[:, np.newaxis] / total_weight  # ordered as the inputs are ordered

    return time, gap_basin, gap_grid


def gldas_grace(gldas='gldas_data.npz', grace='targets.npz'):
    """
    estimate the corr, nrmse and nse between the gldas and GRACE during the training, validating and testing period
    :param gldas:
    :param grace:
    :return:
    """
    # g_time, g_basin, g_grid = gldas2series()
    # np.savez('gldas_data.npz', g_time=g_time, g_basin=g_basin, g_grid=g_grid)
    gldas = np.load(gldas, allow_pickle=True)
    g_grid = gldas['g_grid'][27:, ]  # from 2002-4 to 2021-11
    # remove the gldas data during the data gap period
    g_grid = np.concatenate((g_grid[0:183, ...], g_grid[194:236, ...]), axis=0)
    g_grid = g_grid[:, 103:135, 163:243]

    grace = np.load(grace, allow_pickle=True)
    grace_grid = grace['grace']  # 183*32*80  S*H*W
    gfo_grid = grace['gfo']  # 42*32*80
    G_grid = np.concatenate((grace_grid, gfo_grid), axis=0)

    perf_withoutDA = io.loadmat('perf_withoutDA.mat')
    U_net = perf_withoutDA['basin_result']  # columns=['time', 'model_basin', 'targets_basin','period_flag']
    flag = U_net[:, 3]  # 1, 2,3 represent training, validating and testing, respectively

    tr_g = g_grid[flag == 1]
    tr_G = G_grid[flag == 1]
    val_g = g_grid[flag == 2]
    val_G = G_grid[flag == 2]
    test_g = g_grid[flag == 3]
    test_G = G_grid[flag == 3]

    corr_tr = corr(tr_G, tr_g)
    corr_val = corr(val_G, val_g)
    corr_test = corr(test_G, test_g)

    nse_tr = nse(tr_G, tr_g)
    nse_val = nse(val_G, val_g)
    nse_test = nse(test_G, test_g)

    nrmse_tr = nrmse(tr_G, tr_g)
    nrmse_val = nrmse(val_G, val_g)
    nrmse_test = nrmse(test_G, test_g)

    return corr_tr, corr_val, corr_test, nse_tr, nse_val, nse_test, nrmse_tr, nrmse_val, nrmse_test


def grid2lon_lat_value(grid, output_file='grid.txt'):
    """
    convert the grid data (H*W) to three columns (i.e., [lon, lat, value]) saved in the .txt file
    :param grid: the grid to be converted
    :param output_file: the output file name
    :return:
    """
    # first channel is the latitude while the second is longitude
    lat_lon_grid = lat_lon_gen(resolution=0.5)  # 2*360*720
    lat_lon_grid = lat_lon_grid[:, 103:135, 163:243]
    lon_lat_value = np.empty((32 * 80, 3))
    # set the values outside the region to nan
    grid[mask != 1] = np.nan

    lon_lat_value[:, 0] = lat_lon_grid[1].flatten()  # longitude
    lon_lat_value[:, 1] = lat_lon_grid[0].flatten()  # latitude
    lon_lat_value[:, 2] = grid.flatten()  # value
    np.savetxt(output_file, X=lon_lat_value, fmt='%.2f')
    return lon_lat_value


if __name__ == '__main__':
    perf_withoutDA = io.loadmat('perf_withoutDA.mat')
