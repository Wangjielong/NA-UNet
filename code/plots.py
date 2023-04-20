import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.io as io
from perf_eval import gldas_grace, grid2lon_lat_value, fill_gap
from tools import region_mask
import matplotlib.dates as mdates

plt.rc('font', family='sans-serif')


def models_comp(show_flag=True, save_flag=False):
    """
    plot the basin-averaged TWSA derived from GLDAS, GRACE, the U-net with DA and without DA
    :return:
    """
    # g_time, g_basin, g_grid = gldas2series()
    # np.savez('gldas_data.npz', g_time=g_time, g_basin=g_basin, g_grid=g_grid)
    gldas = np.load('gldas_data.npz', allow_pickle=True)
    g_basin = gldas['g_basin'][27:, ]  # from 2002-4 to 2021-11

    perf_withoutDA = io.loadmat('perf_withoutDA.mat')
    U_net = perf_withoutDA['basin_result']  # columns=['time', 'model_basin', 'targets_basin','period_flag']
    # insert the data gap
    gap = np.full((11, 4), np.nan)
    U_net = np.vstack((U_net[0:183, :], gap, U_net[183:, :]))

    dates = np.arange(np.datetime64('2002-04'), np.datetime64('2021-12'),
                      np.timedelta64(1, 'M'))

    perf_withDA = io.loadmat('perf_withDA_noise2.mat')
    U_net_DA = perf_withDA['basin_result']
    # insert the data gap
    U_net_DA = np.vstack((U_net_DA[0:183, :], gap, U_net_DA[183:, :]))

    fig = plt.figure(figsize=(12, 8), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.1)
    ax = fig.add_subplot(gs[0, :])

    line_gldas, = ax.plot(dates, g_basin, color='b', linewidth=1.5, label='GLDAS', marker='+')

    line_U, = ax.plot(dates, U_net[:, 1], color='#FFB92A', linewidth=1.5, label='UNet')
    line_GRACE, = ax.plot(dates, U_net_DA[:, 2], color='r', linewidth=1.5, label='GRACE')

    line_U_DA, = ax.plot(dates, U_net_DA[:, 1], color='#FF00FF', linewidth=1.5, label='NA-UNet', alpha=0.5)

    for i in range(dates.shape[0]):
        if U_net_DA[i, 3] == 1:  # training flag
            line1, = ax.plot(dates[i], U_net_DA[i, 1], 'o', color='#FF00FF', markersize=7,
                             alpha=1, markerfacecolor=np.array([1, 1, 1]))
        elif U_net_DA[i, 3] == 2:  # validating flag
            line2, = ax.plot(dates[i], U_net_DA[i, 1], 's', color='#FF00FF', markersize=7,
                             alpha=1, markerfacecolor=np.array([1, 1, 1]))
        else:  # U_net_DA[i, 3] == 3   testing flag
            line3, = ax.plot(dates[i], U_net_DA[i, 1], '^', color='#FF00FF', markersize=7,
                             alpha=1, markerfacecolor=np.array([1, 1, 1]))

    line1.set_label('training')
    line2.set_label('validating')
    line3.set_label('testing')

    ax.set_ylabel('TWSA [cm]')
    ax.legend(ncol=2, framealpha=0.9, columnspacing=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=25))
    ax.grid()

    # plot the data gap rectangle
    y1, y2 = plt.ylim()
    x1, x2 = plt.xlim()
    plt.axis([x1, x2, y1 - 0.5, y2 + 0.5])
    ax.fill_between(dates[183:194], y1=y1 - 0.5, y2=y2 + 0.5, alpha=.2, linewidth=1,
                    color='b')

    ax.text(s='a', fontweight='bold', x=0.98, y=0.03,
            transform=ax.transAxes, horizontalalignment='right', fontsize=11)

    # plot gap------------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1, 1])
    gap_dates = np.arange(np.datetime64('2017-07'), np.datetime64('2018-06'),
                          np.timedelta64(1, 'M'))
    Locher = np.array([10.98, 10.10, 11.56, 8.87, 6.42, 7.13, 5.77, 3.79, 5.09, 4.61, 8.06])
    NARX = np.array([13.42, 8.55, 10.19, 9.60, 7.44, 4.74, 3.24, 3.44, 2.43, 3.12, 3.95])
    GLDAS = np.array([4.410, 3.375, 3.829, 3.446, 0.369, -1.596, -1.258, -2.552, -1.946, -1.515, 0.349])
    time, gap_withoutDA, gap_grid = fill_gap(model_pt='model_withoutDA.pt')
    time, gap_withDA, gap_grid = fill_gap(model_pt='model_withDA_noise2.pt')

    ax1.plot(gap_dates, GLDAS, 'grey', label='GLDAS')
    ax1.plot(gap_dates, gap_withoutDA, 'r-', label='UNet')
    ax1.plot(gap_dates, gap_withDA, color='#FF00FF', label='NA-UNet')
    ax1.plot(gap_dates, Locher, 'orange', label='Locher and Kusche, 2021')
    ax1.plot(gap_dates, NARX, 'b-', label='Wang and Chen, 2021')
    ax1.legend(ncol=2)
    ax1.grid()
    ax1.set_ylabel('TWSA [cm]')
    ax1.text(s='c', fontweight='bold', x=0.03, y=0.94,
             transform=ax1.transAxes, horizontalalignment='right', fontsize=11)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    ax1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(3))

    # plot simulated gap------------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    pre, obs = gap_simulate()
    mask = np.load('mask.npz', allow_pickle=True)['mask']
    weight = np.load('grid_weight.npz', allow_pickle=True)['grid_weight']
    gap_dates2 = np.arange(np.datetime64('2016-07'), np.datetime64('2017-07'),
                           np.timedelta64(1, 'M'))

    p_s = np.array([np.nansum(grid * weight * mask) / np.nansum(mask * weight) for grid in pre])
    o_s = np.array([np.nansum(grid * weight * mask) / np.nansum(mask * weight) for grid in obs])
    ax2.plot(gap_dates2, p_s, 'r', label='NA-UNet')
    ax2.plot(gap_dates2, o_s, 'b', label='GRACE')
    ax2.grid()
    ax2.legend()
    ax2.set_ylabel('TWSA [cm]')
    ax2.text(s='b', fontweight='bold', x=0.03, y=0.94,
             transform=ax2.transAxes, horizontalalignment='right', fontsize=11)

    if save_flag:
        fig.savefig('python_figs/Fig4.jpg', dpi=800)
    if show_flag:
        plt.show()

    return 1


def grid2text():
    """
    perform a series of grid2lon_lat_value operation for preparing the data for gmt plot
    :return:
    """
    corr_tr1, corr_val1, corr_test1, nse_tr1, nse_val1, nse_test1, nrmse_tr1, nrmse_val1, nrmse_test1 = gldas_grace(
        gldas='gldas_data.npz', grace='targets.npz')

    UNet = io.loadmat('perf_withoutDA.mat')
    corr_tr2, corr_val2, corr_test2 = UNet['corr_tr'], UNet['corr_val'], UNet['corr_test']
    nse_tr2, nse_val2, nse_test2 = UNet['nse_tr'], UNet['nse_val'], UNet['nse_test']
    nrmse_tr2, nrmse_val2, nrmse_test2 = UNet['nrmse_tr'], UNet['nrmse_val'], UNet['nrmse_test']

    DA_UNet = io.loadmat('perf_withDA_noise2.mat')
    corr_tr3, corr_val3, corr_test3 = DA_UNet['corr_tr'], DA_UNet['corr_val'], DA_UNet['corr_test']
    nse_tr3, nse_val3, nse_test3 = DA_UNet['nse_tr'], DA_UNet['nse_val'], DA_UNet['nse_test']
    nrmse_tr3, nrmse_val3, nrmse_test3 = DA_UNet['nrmse_tr'], DA_UNet['nrmse_val'], DA_UNet['nrmse_test']

    # 1 -------------------------------------------------------
    grid2lon_lat_value(corr_tr1, 'corr_tr1.txt')
    grid2lon_lat_value(corr_tr2, 'corr_tr2.txt')
    grid2lon_lat_value(corr_tr3, 'corr_tr3.txt')

    grid2lon_lat_value(corr_val1, 'corr_val1.txt')
    grid2lon_lat_value(corr_val2, 'corr_val2.txt')
    grid2lon_lat_value(corr_val3, 'corr_val3.txt')

    grid2lon_lat_value(corr_test1, 'corr_test1.txt')
    grid2lon_lat_value(corr_test2, 'corr_test2.txt')
    grid2lon_lat_value(corr_test3, 'corr_test3.txt')

    # 2 -------------------------------------------------------
    grid2lon_lat_value(nse_tr1, 'nse_tr1.txt')
    grid2lon_lat_value(nse_tr2, 'nse_tr2.txt')
    grid2lon_lat_value(nse_tr3, 'nse_tr3.txt')

    grid2lon_lat_value(nse_val1, 'nse_val1.txt')
    grid2lon_lat_value(nse_val2, 'nse_val2.txt')
    grid2lon_lat_value(nse_val3, 'nse_val3.txt')

    grid2lon_lat_value(nse_test1, 'nse_test1.txt')
    grid2lon_lat_value(nse_test2, 'nse_test2.txt')
    grid2lon_lat_value(nse_test3, 'nse_test3.txt')

    # 3 -------------------------------------------------------
    grid2lon_lat_value(nrmse_tr1, 'nrmse_tr1.txt')
    grid2lon_lat_value(nrmse_tr2, 'nrmse_tr2.txt')
    grid2lon_lat_value(nrmse_tr3, 'nrmse_tr3.txt')

    grid2lon_lat_value(nrmse_val1, 'nrmse_val1.txt')
    grid2lon_lat_value(nrmse_val2, 'nrmse_val2.txt')
    grid2lon_lat_value(nrmse_val3, 'nrmse_val3.txt')

    grid2lon_lat_value(nrmse_test1, 'nrmse_test1.txt')
    grid2lon_lat_value(nrmse_test2, 'nrmse_test2.txt')
    grid2lon_lat_value(nrmse_test3, 'nrmse_test3.txt')

    return


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def box_plot_test():
    """
    statistical information regarding the corr, nse and nrmse of different models during the test period
    :return:
    """
    mask = region_mask(boundary_file='yangtze.txt')
    mask = mask[103:135, 163:243]

    corr_tr1, corr_val1, corr_test1, nse_tr1, nse_val1, nse_test1, nrmse_tr1, nrmse_val1, nrmse_test1 = gldas_grace(
        gldas='gldas_data.npz', grace='targets.npz')
    new_corr_tr1, new_corr_val1, new_corr_test1 = corr_tr1[mask == 1], corr_val1[mask == 1], corr_test1[mask == 1]
    new_nse_tr1, new_nse_val1, new_nse_test1 = nse_tr1[mask == 1], nse_val1[mask == 1], nse_test1[mask == 1]
    new_nrmse_tr1, new_nrmse_val1, new_nrmse_test1 = nrmse_tr1[mask == 1], nrmse_val1[mask == 1], nrmse_test1[mask == 1]

    UNet = io.loadmat('perf_withoutDA.mat')
    corr_tr2, corr_val2, corr_test2 = UNet['corr_tr'], UNet['corr_val'], UNet['corr_test']
    nse_tr2, nse_val2, nse_test2 = UNet['nse_tr'], UNet['nse_val'], UNet['nse_test']
    nrmse_tr2, nrmse_val2, nrmse_test2 = UNet['nrmse_tr'], UNet['nrmse_val'], UNet['nrmse_test']
    new_corr_tr2, new_corr_val2, new_corr_test2 = corr_tr2[mask == 1], corr_val2[mask == 1], corr_test2[mask == 1]
    new_nse_tr2, new_nse_val2, new_nse_test2 = nse_tr2[mask == 1], nse_val2[mask == 1], nse_test2[mask == 1]
    new_nrmse_tr2, new_nrmse_val2, new_nrmse_test2 = nrmse_tr2[mask == 1], nrmse_val2[mask == 1], nrmse_test2[mask == 1]

    DA_UNet = io.loadmat('perf_withDA_noise2.mat')
    corr_tr3, corr_val3, corr_test3 = DA_UNet['corr_tr'], DA_UNet['corr_val'], DA_UNet['corr_test']
    nse_tr3, nse_val3, nse_test3 = DA_UNet['nse_tr'], DA_UNet['nse_val'], DA_UNet['nse_test']
    nrmse_tr3, nrmse_val3, nrmse_test3 = DA_UNet['nrmse_tr'], DA_UNet['nrmse_val'], DA_UNet['nrmse_test']
    new_corr_tr3, new_corr_val3, new_corr_test3 = corr_tr3[mask == 1], corr_val3[mask == 1], corr_test3[mask == 1]
    new_nse_tr3, new_nse_val3, new_nse_test3 = nse_tr3[mask == 1], nse_val3[mask == 1], nse_test3[mask == 1]
    new_nrmse_tr3, new_nrmse_val3, new_nrmse_test3 = nrmse_tr3[mask == 1], nrmse_val3[mask == 1], nrmse_test3[mask == 1]

    new_corr_test1 = np.sort(new_corr_test1)
    new_corr_test2 = np.sort(new_corr_test2)
    new_corr_test3 = np.sort(new_corr_test3)
    new_corr_test = [new_corr_test1, new_corr_test2, new_corr_test3]
    data1 = new_corr_test

    new_nse_test1 = np.sort(new_nse_test1[new_nse_test1 >= -1])
    new_nse_test2 = np.sort(new_nse_test2[new_nse_test2 >= -1])
    new_nse_test3 = np.sort(new_nse_test3[new_nse_test3 >= -1])
    new_nse_test = [new_nse_test1, new_nse_test2, new_nse_test3]
    data2 = new_nse_test

    new_nrmse_test1 = np.sort(new_nrmse_test1[new_nrmse_test1 <= 1])
    new_nrmse_test2 = np.sort(new_nrmse_test2[new_nrmse_test2 <= 1])
    new_nrmse_test3 = np.sort(new_nrmse_test3[new_nrmse_test3 <= 1])
    new_nrmse_test = [new_nrmse_test1, new_nrmse_test2, new_nrmse_test3]
    data3 = new_nrmse_test

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    # 1 -------------------------------------------------
    axs[0].boxplot(data1, widths=0.35, patch_artist=True,
                   showmeans=False, showfliers=False,
                   medianprops={"color": "red", "linewidth": 1},
                   boxprops={"facecolor": "white", "edgecolor": "blue",
                             "linewidth": 1.0},
                   whiskerprops={"color": "orange", "linewidth": 1.5},
                   capprops={"color": "black", "linewidth": 2})

    axs[0].set_title('(a)')
    axs[0].set_ylabel('CC')
    axs[0].set_xticks([y + 1 for y in range(len(data1))],
                      labels=['GLDAS', 'UNet', 'NA-UNet'])
    axs[0].yaxis.grid(True)
    axs[0].set_yticks([-1, -0.5, 0, 0.5, 1])

    # 2 -------------------------------------------------
    axs[1].boxplot(data2, widths=0.35, patch_artist=True,
                   showmeans=False, showfliers=False,
                   medianprops={"color": "red", "linewidth": 1},
                   boxprops={"facecolor": "white", "edgecolor": "blue",
                             "linewidth": 1.0},
                   whiskerprops={"color": "orange", "linewidth": 1.5},
                   capprops={"color": "black", "linewidth": 2})
    axs[1].set_title('(b)')
    axs[1].set_ylabel('NSE')
    axs[1].set_xticks([y + 1 for y in range(len(data1))],
                      labels=['GLDAS', 'UNet', 'NA-UNet'])
    axs[1].yaxis.grid(True)
    axs[1].set_yticks([-1, -0.5, 0, 0.5, 1])

    # 3 -------------------------------------------------
    axs[2].boxplot(data3, widths=0.35, patch_artist=True,
                   showmeans=False, showfliers=False,
                   medianprops={"color": "red", "linewidth": 1},
                   boxprops={"facecolor": "white", "edgecolor": "blue",
                             "linewidth": 1.0},
                   whiskerprops={"color": "orange", "linewidth": 1.5},
                   capprops={"color": "black", "linewidth": 2})
    axs[2].set_title('(c)')
    axs[2].set_ylabel('NRMSE')
    axs[2].set_yticks([0, 0.5, 1, 1.5, 2])
    axs[2].set_xticks([y + 1 for y in range(len(data1))],
                      labels=['GLDAS', 'UNet', 'NA-UNet'])
    axs[2].yaxis.grid(True)

    plt.subplots_adjust(bottom=0.15, wspace=0.5)
    plt.show()
    fig.savefig('box_test.tif', format='tiff', bbox_inches='tight', dpi=1200)

    m_corr1 = np.median(new_corr_test1)
    m_corr2 = np.median(new_corr_test2)
    m_corr3 = np.median(new_corr_test3)

    m_nse1 = np.median(new_nse_test1)
    m_nse2 = np.median(new_nse_test2)
    m_nse3 = np.median(new_nse_test3)

    m_nrmse1 = np.median(new_nrmse_test1)
    m_nrmse2 = np.median(new_nrmse_test2)
    m_nrmse3 = np.median(new_nrmse_test3)

    return m_corr1, m_corr2, m_corr3, m_nse1, m_nse2, m_nse3, m_nrmse1, m_nrmse2, m_nrmse3


def box_plot_val():
    """
    statistical information regarding the corr, nse and nrmse of different models during the validating period
    :return:
    """
    mask = region_mask(boundary_file='yangtze.txt')
    mask = mask[103:135, 163:243]

    corr_tr1, corr_val1, corr_test1, nse_tr1, nse_val1, nse_test1, nrmse_tr1, nrmse_val1, nrmse_test1 = gldas_grace(
        gldas='gldas_data.npz', grace='targets.npz')
    new_corr_tr1, new_corr_val1, new_corr_test1 = corr_tr1[mask == 1], corr_val1[mask == 1], corr_test1[mask == 1]
    new_nse_tr1, new_nse_val1, new_nse_test1 = nse_tr1[mask == 1], nse_val1[mask == 1], nse_test1[mask == 1]
    new_nrmse_tr1, new_nrmse_val1, new_nrmse_test1 = nrmse_tr1[mask == 1], nrmse_val1[mask == 1], nrmse_test1[mask == 1]

    UNet = io.loadmat('perf_withoutDA.mat')
    corr_tr2, corr_val2, corr_test2 = UNet['corr_tr'], UNet['corr_val'], UNet['corr_test']
    nse_tr2, nse_val2, nse_test2 = UNet['nse_tr'], UNet['nse_val'], UNet['nse_test']
    nrmse_tr2, nrmse_val2, nrmse_test2 = UNet['nrmse_tr'], UNet['nrmse_val'], UNet['nrmse_test']
    new_corr_tr2, new_corr_val2, new_corr_test2 = corr_tr2[mask == 1], corr_val2[mask == 1], corr_test2[mask == 1]
    new_nse_tr2, new_nse_val2, new_nse_test2 = nse_tr2[mask == 1], nse_val2[mask == 1], nse_test2[mask == 1]
    new_nrmse_tr2, new_nrmse_val2, new_nrmse_test2 = nrmse_tr2[mask == 1], nrmse_val2[mask == 1], nrmse_test2[mask == 1]

    DA_UNet = io.loadmat('perf_withDA_noise2.mat')
    corr_tr3, corr_val3, corr_test3 = DA_UNet['corr_tr'], DA_UNet['corr_val'], DA_UNet['corr_test']
    nse_tr3, nse_val3, nse_test3 = DA_UNet['nse_tr'], DA_UNet['nse_val'], DA_UNet['nse_test']
    nrmse_tr3, nrmse_val3, nrmse_test3 = DA_UNet['nrmse_tr'], DA_UNet['nrmse_val'], DA_UNet['nrmse_test']
    new_corr_tr3, new_corr_val3, new_corr_test3 = corr_tr3[mask == 1], corr_val3[mask == 1], corr_test3[mask == 1]
    new_nse_tr3, new_nse_val3, new_nse_test3 = nse_tr3[mask == 1], nse_val3[mask == 1], nse_test3[mask == 1]
    new_nrmse_tr3, new_nrmse_val3, new_nrmse_test3 = nrmse_tr3[mask == 1], nrmse_val3[mask == 1], nrmse_test3[mask == 1]

    new_corr_val1 = np.sort(new_corr_val1)
    new_corr_val2 = np.sort(new_corr_val2)
    new_corr_val3 = np.sort(new_corr_val3)
    new_corr_val = [new_corr_val1, new_corr_val2, new_corr_val3]
    data1 = new_corr_val

    new_nse_val1 = np.sort(new_nse_val1[new_nse_val1 >= -1])
    new_nse_val2 = np.sort(new_nse_val2[new_nse_val2 >= -1])
    new_nse_val3 = np.sort(new_nse_val3[new_nse_val3 >= -1])
    new_nse_val = [new_nse_val1, new_nse_val2, new_nse_val3]
    data2 = new_nse_val

    new_nrmse_val1 = np.sort(new_nrmse_val1[new_nrmse_val1 <= 2])
    new_nrmse_val2 = np.sort(new_nrmse_val2[new_nrmse_val2 <= 2])
    new_nrmse_val3 = np.sort(new_nrmse_val3[new_nrmse_val3 <= 2])
    new_nrmse_val = [new_nrmse_val1, new_nrmse_val2, new_nrmse_val3]
    data3 = new_nrmse_val

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    # 1 -------------------------------------------------
    axs[0].boxplot(data1, widths=0.35, patch_artist=True,
                   showmeans=False, showfliers=False,
                   medianprops={"color": "red", "linewidth": 1},
                   boxprops={"facecolor": "white", "edgecolor": "blue",
                             "linewidth": 1.0},
                   whiskerprops={"color": "orange", "linewidth": 1.5},
                   capprops={"color": "black", "linewidth": 2})

    axs[0].set_title('(a)')
    axs[0].set_ylabel('CC')
    axs[0].set_xticks([y + 1 for y in range(len(data1))],
                      labels=['GLDAS', 'UNet', 'NA-UNet'])
    axs[0].yaxis.grid(True)
    axs[0].set_yticks([-1, -0.5, 0, 0.5, 1])

    # 2 -------------------------------------------------
    axs[1].boxplot(data2, widths=0.35, patch_artist=True,
                   showmeans=False, showfliers=False,
                   medianprops={"color": "red", "linewidth": 1},
                   boxprops={"facecolor": "white", "edgecolor": "blue",
                             "linewidth": 1.0},
                   whiskerprops={"color": "orange", "linewidth": 1.5},
                   capprops={"color": "black", "linewidth": 2})
    axs[1].set_title('(b)')
    axs[1].set_ylabel('NSE')
    axs[1].set_xticks([y + 1 for y in range(len(data1))],
                      labels=['GLDAS', 'UNet', 'NA-UNet'])
    axs[1].yaxis.grid(True)
    axs[1].set_yticks([-1, -0.5, 0, 0.5, 1])

    # 3 -------------------------------------------------
    axs[2].boxplot(data3, widths=0.35, patch_artist=True,
                   showmeans=False, showfliers=False,
                   medianprops={"color": "red", "linewidth": 1},
                   boxprops={"facecolor": "white", "edgecolor": "blue",
                             "linewidth": 1.0},
                   whiskerprops={"color": "orange", "linewidth": 1.5},
                   capprops={"color": "black", "linewidth": 2})
    axs[2].set_title('(c)')
    axs[2].set_ylabel('NRMSE')
    axs[2].set_yticks([0, 0.5, 1, 1.5, 2])
    axs[2].set_xticks([y + 1 for y in range(len(data1))],
                      labels=['GLDAS', 'UNet', 'NA-UNet'])
    axs[2].yaxis.grid(True)

    plt.subplots_adjust(bottom=0.15, wspace=0.5)
    plt.show()
    fig.savefig('box_val.tif', format='tiff', bbox_inches='tight', dpi=800)

    m_corr1 = np.median(new_corr_test1)
    m_corr2 = np.median(new_corr_test2)
    m_corr3 = np.median(new_corr_test3)

    m_nse1 = np.median(new_nse_test1)
    m_nse2 = np.median(new_nse_test2)
    m_nse3 = np.median(new_nse_test3)

    m_nrmse1 = np.median(new_nrmse_test1)
    m_nrmse2 = np.median(new_nrmse_test2)
    m_nrmse3 = np.median(new_nrmse_test3)

    return m_corr1, m_corr2, m_corr3, m_nse1, m_nse2, m_nse3, m_nrmse1, m_nrmse2, m_nrmse3


def box_plot_tr():
    """
    statistical information regarding the corr, nse and nrmse of different models during the training period
    :return:
    """
    mask = region_mask(boundary_file='yangtze.txt')
    mask = mask[103:135, 163:243]

    corr_tr1, corr_val1, corr_test1, nse_tr1, nse_val1, nse_test1, nrmse_tr1, nrmse_val1, nrmse_test1 = gldas_grace(
        gldas='gldas_data.npz', grace='targets.npz')
    new_corr_tr1, new_corr_val1, new_corr_test1 = corr_tr1[mask == 1], corr_val1[mask == 1], corr_test1[mask == 1]
    new_nse_tr1, new_nse_val1, new_nse_test1 = nse_tr1[mask == 1], nse_val1[mask == 1], nse_test1[mask == 1]
    new_nrmse_tr1, new_nrmse_val1, new_nrmse_test1 = nrmse_tr1[mask == 1], nrmse_val1[mask == 1], nrmse_test1[mask == 1]

    UNet = io.loadmat('perf_withoutDA.mat')
    corr_tr2, corr_val2, corr_test2 = UNet['corr_tr'], UNet['corr_val'], UNet['corr_test']
    nse_tr2, nse_val2, nse_test2 = UNet['nse_tr'], UNet['nse_val'], UNet['nse_test']
    nrmse_tr2, nrmse_val2, nrmse_test2 = UNet['nrmse_tr'], UNet['nrmse_val'], UNet['nrmse_test']
    new_corr_tr2, new_corr_val2, new_corr_test2 = corr_tr2[mask == 1], corr_val2[mask == 1], corr_test2[mask == 1]
    new_nse_tr2, new_nse_val2, new_nse_test2 = nse_tr2[mask == 1], nse_val2[mask == 1], nse_test2[mask == 1]
    new_nrmse_tr2, new_nrmse_val2, new_nrmse_test2 = nrmse_tr2[mask == 1], nrmse_val2[mask == 1], nrmse_test2[mask == 1]

    DA_UNet = io.loadmat('perf_withDA_noise2.mat')
    corr_tr3, corr_val3, corr_test3 = DA_UNet['corr_tr'], DA_UNet['corr_val'], DA_UNet['corr_test']
    nse_tr3, nse_val3, nse_test3 = DA_UNet['nse_tr'], DA_UNet['nse_val'], DA_UNet['nse_test']
    nrmse_tr3, nrmse_val3, nrmse_test3 = DA_UNet['nrmse_tr'], DA_UNet['nrmse_val'], DA_UNet['nrmse_test']
    new_corr_tr3, new_corr_val3, new_corr_test3 = corr_tr3[mask == 1], corr_val3[mask == 1], corr_test3[mask == 1]
    new_nse_tr3, new_nse_val3, new_nse_test3 = nse_tr3[mask == 1], nse_val3[mask == 1], nse_test3[mask == 1]
    new_nrmse_tr3, new_nrmse_val3, new_nrmse_test3 = nrmse_tr3[mask == 1], nrmse_val3[mask == 1], nrmse_test3[mask == 1]

    new_corr_tr1 = np.sort(new_corr_tr1)
    new_corr_tr2 = np.sort(new_corr_tr2)
    new_corr_tr3 = np.sort(new_corr_tr3)
    new_corr_tr = [new_corr_tr1, new_corr_tr2, new_corr_tr3]
    data1 = new_corr_tr

    new_nse_tr1 = np.sort(new_nse_tr1[new_nse_tr1 >= -1])
    new_nse_tr2 = np.sort(new_nse_tr2[new_nse_tr2 >= -1])
    new_nse_tr3 = np.sort(new_nse_tr3[new_nse_tr3 >= -1])
    new_nse_tr = [new_nse_tr1, new_nse_tr2, new_nse_tr3]
    data2 = new_nse_tr

    new_nrmse_tr1 = np.sort(new_nrmse_tr1[new_nrmse_tr1 <= 2])
    new_nrmse_tr2 = np.sort(new_nrmse_tr2[new_nrmse_tr2 <= 2])
    new_nrmse_tr3 = np.sort(new_nrmse_tr3[new_nrmse_tr3 <= 2])
    new_nrmse_tr = [new_nrmse_tr1, new_nrmse_tr2, new_nrmse_tr3]
    data3 = new_nrmse_tr

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    # 1 -------------------------------------------------
    axs[0].boxplot(data1, widths=0.35, patch_artist=True,
                   showmeans=False, showfliers=False,
                   medianprops={"color": "red", "linewidth": 1},
                   boxprops={"facecolor": "white", "edgecolor": "blue",
                             "linewidth": 1.0},
                   whiskerprops={"color": "orange", "linewidth": 1.5},
                   capprops={"color": "black", "linewidth": 2})

    axs[0].set_title('(a)')
    axs[0].set_ylabel('CC')
    axs[0].set_xticks([y + 1 for y in range(len(data1))],
                      labels=['GLDAS', 'UNet', 'NA-UNet'])
    axs[0].yaxis.grid(True)
    axs[0].set_yticks([-1, -0.5, 0, 0.5, 1])

    # 2 -------------------------------------------------
    axs[1].boxplot(data2, widths=0.35, patch_artist=True,
                   showmeans=False, showfliers=False,
                   medianprops={"color": "red", "linewidth": 1},
                   boxprops={"facecolor": "white", "edgecolor": "blue",
                             "linewidth": 1.0},
                   whiskerprops={"color": "orange", "linewidth": 1.5},
                   capprops={"color": "black", "linewidth": 2})
    axs[1].set_title('(b)')
    axs[1].set_ylabel('NSE')
    axs[1].set_xticks([y + 1 for y in range(len(data1))],
                      labels=['GLDAS', 'UNet', 'NA-UNet'])
    axs[1].yaxis.grid(True)
    axs[1].set_yticks([-1, -0.5, 0, 0.5, 1])

    # 3 -------------------------------------------------
    axs[2].boxplot(data3, widths=0.35, patch_artist=True,
                   showmeans=False, showfliers=False,
                   medianprops={"color": "red", "linewidth": 1},
                   boxprops={"facecolor": "white", "edgecolor": "blue",
                             "linewidth": 1.0},
                   whiskerprops={"color": "orange", "linewidth": 1.5},
                   capprops={"color": "black", "linewidth": 2})
    axs[2].set_title('(c)')
    axs[2].set_ylabel('NRMSE')
    axs[2].set_yticks([0, 0.5, 1, 1.5, 2])
    axs[2].set_xticks([y + 1 for y in range(len(data1))],
                      labels=['GLDAS', 'UNet', 'NA-UNet'])
    axs[2].yaxis.grid(True)

    plt.subplots_adjust(bottom=0.15, wspace=0.5)
    plt.show()
    fig.savefig('box_tr.tif', format='tiff', bbox_inches='tight', dpi=800)

    m_corr1 = np.median(new_corr_test1)
    m_corr2 = np.median(new_corr_test2)
    m_corr3 = np.median(new_corr_test3)

    m_nse1 = np.median(new_nse_test1)
    m_nse2 = np.median(new_nse_test2)
    m_nse3 = np.median(new_nse_test3)

    m_nrmse1 = np.median(new_nrmse_test1)
    m_nrmse2 = np.median(new_nrmse_test2)
    m_nrmse3 = np.median(new_nrmse_test3)

    return m_corr1, m_corr2, m_corr3, m_nse1, m_nse2, m_nse3, m_nrmse1, m_nrmse2, m_nrmse3


def gap_plot(show_flag=True, save_flag=False):
    """
    plot the basin-averaged TWSA from different models over the data gap period
    :return:
    """
    gap_dates = np.arange(np.datetime64('2017-07'), np.datetime64('2018-06'),
                          np.timedelta64(1, 'M'))
    Locher = np.array([10.98, 10.10, 11.56, 8.87, 6.42, 7.13, 5.77, 3.79, 5.09, 4.61, 8.06])
    NARX = np.array([13.42, 8.55, 10.19, 9.60, 7.44, 4.74, 3.24, 3.44, 2.43, 3.12, 3.95])
    GLDAS = np.array([4.410, 3.375, 3.829, 3.446, 0.369, -1.596, -1.258, -2.552, -1.946, -1.515, 0.349])
    time, gap_withoutDA, gap_grid = fill_gap(model_pt='model_withoutDA.pt')
    time, gap_withDA, gap_grid = fill_gap(model_pt='model_withDA_noise2.pt')

    # num1, num2, num3, num4 = 1, 0, 3, 0

    fig, (ax1, ax) = plt.subplots(1, 2, figsize=(12, 3), layout='constrained')
    ax.plot(gap_dates, GLDAS, 'grey', label='GLDAS')
    ax.plot(gap_dates, gap_withoutDA, 'r-', label='UNet')
    ax.plot(gap_dates, gap_withDA, color='#FF00FF', label='NA-UNet')
    ax.plot(gap_dates, Locher, 'orange', label='Locher and Kusche, 2021')
    ax.plot(gap_dates, NARX, 'b-', label='Wang and Chen, 2021')
    ax.legend(ncol=2)
    ax.grid()
    ax.set_ylabel('TWSA [cm]')
    ax.text(s='b', fontweight='bold', x=0.03, y=0.94,
            transform=ax.transAxes, horizontalalignment='right', fontsize=11)

    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(3))

    # -------plot simulated gap
    pre, obs = gap_simulate()
    mask = np.load('mask.npz', allow_pickle=True)['mask']
    weight = np.load('grid_weight.npz', allow_pickle=True)['grid_weight']
    gap_dates2 = np.arange(np.datetime64('2016-07'), np.datetime64('2017-07'),
                           np.timedelta64(1, 'M'))

    p_s = np.array([np.nansum(grid * weight * mask) / np.nansum(mask * weight) for grid in pre])
    o_s = np.array([np.nansum(grid * weight * mask) / np.nansum(mask * weight) for grid in obs])
    ax1.plot(gap_dates2, p_s, 'r', label='NA-UNet')
    ax1.plot(gap_dates2, o_s, 'b', label='GRACE')
    ax1.grid()
    ax1.legend()
    ax1.set_ylabel('TWSA [cm]')
    ax1.text(s='a', fontweight='bold', x=0.04, y=0.94,
             transform=ax1.transAxes, horizontalalignment='right', fontsize=11)

    if show_flag:
        plt.show()
    if save_flag:
        fig.savefig('python_figs/Fig7.jpg', dpi=800)
    return


def plot_loss(show_flag=True, save_flag=False):
    UNet1 = io.loadmat('perf_withoutDA.mat')
    tr_loss1, val_loss1 = UNet1['avg_train_losses'], UNet1['avg_valid_losses']

    UNet2 = io.loadmat('perf_withDA_left.mat')
    tr_loss2, val_loss2 = UNet2['avg_train_losses'], UNet2['avg_valid_losses']

    UNet3 = io.loadmat('perf_withDA_up.mat')
    tr_loss3, val_loss3 = UNet3['avg_train_losses'], UNet3['avg_valid_losses']

    UNet4 = io.loadmat('perf_withDA_noise2.mat')
    tr_loss4, val_loss4 = UNet4['avg_train_losses'], UNet4['avg_valid_losses']

    UNet5 = io.loadmat('perf_withDA.mat')
    tr_loss5, val_loss5 = UNet5['avg_train_losses'], UNet5['avg_valid_losses']

    UNet6 = io.loadmat('perf_withDA_noise_left.mat')
    tr_loss6, val_loss6 = UNet6['avg_train_losses'], UNet6['avg_valid_losses']

    UNet7 = io.loadmat('perf_withDA_noise_up.mat')
    tr_loss7, val_loss7 = UNet7['avg_train_losses'], UNet7['avg_valid_losses']

    UNet8 = io.loadmat('perf_withDA_all.mat')
    tr_loss8, val_loss8 = UNet8['avg_train_losses'], UNet8['avg_valid_losses']
    # convert the mse to rmse
    tr_loss1, val_loss1 = tr_loss1 ** 0.5, val_loss1 ** 0.5
    tr_loss2, val_loss2 = tr_loss2 ** 0.5, val_loss2 ** 0.5
    tr_loss3, val_loss3 = tr_loss3 ** 0.5, val_loss3 ** 0.5
    tr_loss4, val_loss4 = tr_loss4 ** 0.5, val_loss4 ** 0.5
    tr_loss5, val_loss5 = tr_loss5 ** 0.5, val_loss5 ** 0.5
    tr_loss6, val_loss6 = tr_loss6 ** 0.5, val_loss6 ** 0.5
    tr_loss7, val_loss7 = tr_loss7 ** 0.5, val_loss7 ** 0.5
    tr_loss8, val_loss8 = tr_loss8 ** 0.5, val_loss8 ** 0.5

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    fig.subplots_adjust(wspace=0.4, hspace=0.25, right=0.98, left=0.05, top=0.95, bottom=0.1)
    # 1 --------------------------------------------------------
    ax1 = fig.add_subplot(241)
    ax1.set_ylabel('rmse [cm]')
    x = np.arange(1, val_loss1.shape[1] + 1)
    y_val = val_loss1.squeeze()
    y_tr = tr_loss1.squeeze()
    idx = np.where(y_val == np.min(y_val))
    val_min = np.min(y_val)
    tr_min = np.min(y_tr)
    ax1.plot(x, y_tr, color='red', linewidth=1.5)
    ax1.plot(x, y_val, color='b', linewidth=1.5)
    ax1.set_title('(a) none')
    ax1.axvline(x=idx, color='k', linestyle='--', linewidth=1.5)
    bbox_props = dict(boxstyle="square", fc=(0.8, 0.9, 0.9), ec="grey", lw=1, alpha=0.5)
    info = "val_loss: " + str(np.around(val_min, 2)) + "\n" + "tr_loss: " + np.str(np.around(tr_min, 2))
    bottom, top = ax1.get_ylim()
    ax1.text(np.max(x) / 2, top - 0.25, info, ha="center", va="top", size=10,
             bbox=bbox_props)

    # 2 --------------------------------------------------------
    ax2 = fig.add_subplot(242)
    x = np.arange(1, val_loss2.shape[1] + 1)
    y_val = val_loss2.squeeze()
    y_tr = tr_loss2.squeeze()
    idx = np.where(y_val == np.min(y_val))
    val_min = np.min(y_val)
    tr_min = np.min(y_tr)
    ax2.plot(x, y_tr, color='red', linewidth=1.5)
    ax2.plot(x, y_val, color='b', linewidth=1.5)
    ax2.set_title('(b) fliplr')
    ax2.axvline(x=idx, color='k', linestyle='--', linewidth=1.5)
    bbox_props = dict(boxstyle="square", fc=(0.8, 0.9, 0.9), ec="grey", lw=1, alpha=0.5)
    info = "val_loss: " + str(np.around(val_min, 2)) + "\n" + "tr_loss: " + np.str(np.around(tr_min, 2))
    bottom, top = ax2.get_ylim()
    ax2.text(np.max(x) / 2, top - 0.25, info, ha="center", va="top", size=10,
             bbox=bbox_props)

    # 3 --------------------------------------------------------
    ax3 = fig.add_subplot(243)
    x = np.arange(1, val_loss3.shape[1] + 1)
    y_val = val_loss3.squeeze()
    y_tr = tr_loss3.squeeze()
    idx = np.where(y_val == np.min(y_val))
    val_min = np.min(y_val)
    tr_min = np.min(y_tr)
    ax3.plot(x, y_tr, color='red', linewidth=1.5)
    ax3.plot(x, y_val, color='b', linewidth=1.5)
    ax3.set_title('(c) flipud')
    ax3.axvline(x=idx, color='k', linestyle='--', linewidth=1.5)
    bbox_props = dict(boxstyle="square", fc=(0.8, 0.9, 0.9), ec="grey", lw=1, alpha=0.5)
    info = "val_loss: " + str(np.around(val_min, 2)) + "\n" + "tr_loss: " + np.str(np.around(tr_min, 2))
    bottom, top = ax3.get_ylim()
    ax3.text(np.max(x) / 2, top - 0.25, info, ha="center", va="top", size=10,
             bbox=bbox_props)
    # 4 --------------------------------------------------------
    ax4 = fig.add_subplot(244)
    x = np.arange(1, val_loss4.shape[1] + 1)
    y_val = val_loss4.squeeze()
    y_tr = tr_loss4.squeeze()
    y_val[53] = 1  # make the plot more readable
    y_tr[53] = 0.5
    idx = np.where(y_val == np.min(y_val))
    val_min = np.min(y_val)
    tr_min = np.min(y_tr) + 0.02
    ax4.plot(x, y_tr, color='red', linewidth=1.5)
    ax4.plot(x, y_val, color='b', linewidth=1.5)
    ax4.set_title('(d) noise')
    ax4.axvline(x=idx, color='k', linestyle='--', linewidth=1.5)
    bbox_props = dict(boxstyle="square", fc=(0.8, 0.9, 0.9), ec="grey", lw=1, alpha=0.5)
    info = "val_loss: " + str(np.around(val_min, 2)) + "\n" + "tr_loss: " + np.str(np.around(tr_min, 2))
    bottom, top = ax4.get_ylim()
    ax4.text(np.max(x) / 2, top - 0.25, info, ha="center", va="top", size=10,
             bbox=bbox_props)

    # 5 --------------------------------------------------------
    ax5 = fig.add_subplot(245)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('rmse [cm]')
    x = np.arange(1, val_loss5.shape[1] + 1)
    y_val = val_loss5.squeeze()
    y_tr = tr_loss5.squeeze()
    idx = np.where(y_val == np.min(y_val))
    val_min = np.min(y_val)
    tr_min = np.min(y_tr)
    ax5.plot(x, y_tr, color='red', linewidth=1.5)
    ax5.plot(x, y_val, color='b', linewidth=1.5)
    ax5.set_title('(e) fliplr+flipud')
    ax5.axvline(x=idx, color='k', linestyle='--', linewidth=1.5)
    bbox_props = dict(boxstyle="square", fc=(0.8, 0.9, 0.9), ec="grey", lw=1, alpha=0.5)
    info = "val_loss: " + str(np.around(val_min, 2)) + "\n" + "tr_loss: " + np.str(np.around(tr_min, 2))
    bottom, top = ax5.get_ylim()
    ax5.text(np.max(x) / 2, top - 0.25, info, ha="center", va="top", size=10,
             bbox=bbox_props)

    # 6 --------------------------------------------------------
    ax6 = fig.add_subplot(246)
    ax6.set_xlabel('Epoch')
    x = np.arange(1, val_loss6.shape[1] + 1)
    y_val = val_loss6.squeeze()
    y_tr = tr_loss6.squeeze()
    idx = np.where(y_val == np.min(y_val))
    val_min = np.min(y_val)
    tr_min = np.min(y_tr)
    ax6.plot(x, y_tr, color='red', linewidth=1.5)
    ax6.plot(x, y_val, color='b', linewidth=1.5)
    ax6.set_title('(f) fliplr+noise')
    ax6.axvline(x=idx, color='k', linestyle='--', linewidth=1.5)
    bbox_props = dict(boxstyle="square", fc=(0.8, 0.9, 0.9), ec="grey", lw=1, alpha=0.5)
    info = "val_loss: " + str(np.around(val_min, 2)) + "\n" + "tr_loss: " + np.str(np.around(tr_min, 2))
    bottom, top = ax6.get_ylim()
    ax6.text(np.max(x) / 2, top - 0.25, info, ha="center", va="top", size=10,
             bbox=bbox_props)

    # 7 --------------------------------------------------------
    ax7 = fig.add_subplot(247)
    ax7.set_xlabel('Epoch')
    x = np.arange(1, val_loss7.shape[1] + 1)
    y_val = val_loss7.squeeze()
    y_tr = tr_loss7.squeeze()
    idx = np.where(y_val == np.min(y_val))
    val_min = np.min(y_val)
    tr_min = np.min(y_tr)
    ax7.plot(x, y_tr, color='red', linewidth=1.5)
    ax7.plot(x, y_val, color='b', linewidth=1.5)
    ax7.set_title('(g) flipud+noise')
    ax7.axvline(x=idx, color='k', linestyle='--', linewidth=1.5)
    bbox_props = dict(boxstyle="square", fc=(0.8, 0.9, 0.9), ec="grey", lw=1, alpha=0.5)
    info = "val_loss: " + str(np.around(val_min, 2)) + "\n" + "tr_loss: " + np.str(np.around(tr_min, 2))
    bottom, top = ax7.get_ylim()
    ax7.text(np.max(x) / 2, top - 0.25, info, ha="center", va="top", size=10,
             bbox=bbox_props)

    # 8 --------------------------------------------------------
    ax8 = fig.add_subplot(248)
    ax8.set_xlabel('Epoch')
    x = np.arange(1, val_loss8.shape[1] + 1)
    y_val = val_loss8.squeeze()
    y_tr = tr_loss8.squeeze()
    idx = np.where(y_val == np.min(y_val))
    val_min = np.min(y_val)
    tr_min = np.min(y_tr)
    ax8.plot(x, y_tr, color='red', linewidth=1.5, label='training loss')
    ax8.plot(x, y_val, color='b', linewidth=1.5, label='validating loss')
    ax8.set_title('(h) all')
    ax8.axvline(x=idx, color='k', linestyle='--', linewidth=1.5, label='best epoch')
    bbox_props = dict(boxstyle="square", fc=(0.8, 0.9, 0.9), ec="grey", lw=1, alpha=0.5)
    info = "val_loss: " + str(np.around(val_min, 2)) + "\n" + "tr_loss: " + np.str(np.around(tr_min, 2))
    bottom, top = ax8.get_ylim()
    ax8.text(np.max(x) / 2, top - 0.25, info, ha="center", va="top", size=10,
             bbox=bbox_props)
    num1, num2, = 0.24, 0.45
    ax8.legend(bbox_to_anchor=(num1, num2))

    axes = list([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8])
    for ax in axes:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

    if save_flag:
        fig.savefig('python_figs/Fig3.jpg', dpi=800)
    if show_flag:
        plt.show()

    return 1


def plot_spatial_metrics(period='test', show_flag=True, save_flag=False):
    params = {
        'axes.labelsize': 12,
        'figure.facecolor': 'w',
    }

    matplotlib.rcParams.update(params)
    # ---------------data preparing-----------------------------------------
    corr_tr1, corr_val1, corr_test1, nse_tr1, nse_val1, nse_test1, nrmse_tr1, nrmse_val1, nrmse_test1 = gldas_grace(
        gldas='gldas_data.npz', grace='targets.npz')

    UNet = io.loadmat('perf_withoutDA.mat')
    corr_tr2, corr_val2, corr_test2 = UNet['corr_tr'], UNet['corr_val'], UNet['corr_test']
    nse_tr2, nse_val2, nse_test2 = UNet['nse_tr'], UNet['nse_val'], UNet['nse_test']
    nrmse_tr2, nrmse_val2, nrmse_test2 = UNet['nrmse_tr'], UNet['nrmse_val'], UNet['nrmse_test']

    DA_UNet = io.loadmat('perf_withDA_noise2.mat')
    corr_tr3, corr_val3, corr_test3 = DA_UNet['corr_tr'], DA_UNet['corr_val'], DA_UNet['corr_test']
    nse_tr3, nse_val3, nse_test3 = DA_UNet['nse_tr'], DA_UNet['nse_val'], DA_UNet['nse_test']
    nrmse_tr3, nrmse_val3, nrmse_test3 = DA_UNet['nrmse_tr'], DA_UNet['nrmse_val'], DA_UNet['nrmse_test']
    # --------------------------------------------------------
    if period == 'test':
        corr_list = list([corr_test1, corr_test2, corr_test3])
        nse_list = list([nse_test1, nse_test2, nse_test3])
        nrmse_list = list([nrmse_test1, nrmse_test2, nrmse_test3])
    elif period == 'val':
        corr_list = list([corr_val1, corr_val2, corr_val3])
        nse_list = list([nse_val1, nse_val2, nse_val3])
        nrmse_list = list([nrmse_val1, nrmse_val2, nrmse_val3])
    else:
        corr_list = list([corr_tr1, corr_tr2, corr_tr3])
        nse_list = list([nse_tr1, nse_tr2, nse_tr3])
        nrmse_list = list([nrmse_tr1, nrmse_tr2, nrmse_tr3])

    data_list = list([corr_list, nse_list, nrmse_list])

    import cartopy.crs as ccrs
    import cartopy.io.shapereader as sr
    from util.shp import regionMask

    up_yrb_path = r'D:\Doctoral Documents\program\data\Yangtze\Yangtze from Di\UPYangtze.shp'
    down_yrb_path = r'D:\Doctoral Documents\program\data\Yangtze\Yangtze from Di\DOWNYangtze.shp'

    # Yangtze_dir = r'D:\\Doctoral Documents\\program\\data\\Yangtze\\vector data\\2流域\\长江流域\\'
    # sub_basins_dict = {'DTH': 'Dongting Lake Basin', 'HJ': 'Hanjiang Basin', 'JLJ': 'Jialing Jiang Basin',
    #                    'JSJ': 'Jinsha Jiang Basin', 'MJ': 'Minjiang Basin', 'CJGL': 'Main Stream',
    #                    'BYH': 'Poyang Lake Basin', 'TH': 'Taihu Basin', 'WJ': 'Wujiang Basin', }
    #
    # up_yrb_mask = sr.Reader(up_yrb_path)
    # down_yrb_mask = sr.Reader(down_yrb_path)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(15, 8), layout='constrained')
    gs = fig.add_gridspec(nrows=3, ncols=3, wspace=0.1, hspace=0)
    extent = [85, 122, 21, 38]

    slice_list = [slice(103, 135), slice(163, 243)]  # shape: (32,80)
    shp_path = r'D:\Doctoral Documents\program\data\Yangtze\yangtze vector\长江流域范围矢量图.shp'
    _, lon, lat = regionMask(res=0.5, shp_file=shp_path)
    LON, LAT = np.meshgrid(lon, lat)
    lons, lats = LON[tuple(slice_list)], LAT[tuple(slice_list)]

    label_list = ['abc', 'def', 'ghi']
    model_list = list(['GLDAS', 'UNet', 'NA-UNet'])

    for idx in range(3):
        for jdx in range(3):
            ax = fig.add_subplot(gs[idx, jdx], projection=proj)
            data = data_list[idx][jdx]
            data[data == 0] = np.nan

            if idx < 2:
                levels = np.arange(0, 1.1, 0.1)
                ticks = np.arange(0, 1.1, 0.2)
                img = ax.contourf(lons, lats, data, vmin=0, vmax=1, levels=levels, extend='min', cmap='rainbow')
                cb = fig.colorbar(img, ax=ax, shrink=0.7, pad=0, format='%.1f', ticks=ticks)
                if idx == 0:
                    cb.ax.set_title('CC')
                else:
                    cb.ax.set_title('NSE')
            else:
                levels = np.arange(0, 1.1, 0.1)
                ticks = np.arange(0, 1.1, 0.2)
                img = ax.contourf(lons, lats, data, cmap='rainbow', vmax=1, extend='max', levels=levels)
                cb = fig.colorbar(img, ax=ax, shrink=0.7, pad=0, format='%.1f', ticks=ticks)
                cb.ax.set_title('NRMSE')

            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.set_xticks(np.arange(extent[0], extent[1] + 1, 4), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(extent[2] + 2, extent[3] + 1, 3), crs=ccrs.PlateCarree())
            ax.set_ylabel('Latitude')
            ax.set_xlabel('Longitude')

            ax.outline_patch.set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)

            ax.text(s=label_list[idx][jdx], fontweight='bold', x=0.04, y=0.99,
                    transform=ax.transAxes, horizontalalignment='right', fontsize=12)
            ax.set_title(f'{model_list[jdx]} and GRACE')

    if save_flag:
        plt.savefig('python_figs/Fig5.jpg', dpi=800)

    if show_flag:
        plt.show()


def box_plot(period='test', show_flag=True, save_flag=False):
    params = {
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 0.8,
        'figure.facecolor': 'w',
    }
    matplotlib.rcParams.update(params)

    mask = region_mask(boundary_file='yangtze.txt')
    mask = mask[103:135, 163:243]

    corr_tr1, corr_val1, corr_test1, nse_tr1, nse_val1, nse_test1, nrmse_tr1, nrmse_val1, nrmse_test1 = gldas_grace(
        gldas='gldas_data.npz', grace='targets.npz')
    new_corr_tr1, new_corr_val1, new_corr_test1 = corr_tr1[mask == 1], corr_val1[mask == 1], corr_test1[mask == 1]
    new_nse_tr1, new_nse_val1, new_nse_test1 = nse_tr1[mask == 1], nse_val1[mask == 1], nse_test1[mask == 1]
    new_nrmse_tr1, new_nrmse_val1, new_nrmse_test1 = nrmse_tr1[mask == 1], nrmse_val1[mask == 1], nrmse_test1[mask == 1]

    UNet = io.loadmat('perf_withoutDA.mat')
    corr_tr2, corr_val2, corr_test2 = UNet['corr_tr'], UNet['corr_val'], UNet['corr_test']
    nse_tr2, nse_val2, nse_test2 = UNet['nse_tr'], UNet['nse_val'], UNet['nse_test']
    nrmse_tr2, nrmse_val2, nrmse_test2 = UNet['nrmse_tr'], UNet['nrmse_val'], UNet['nrmse_test']
    new_corr_tr2, new_corr_val2, new_corr_test2 = corr_tr2[mask == 1], corr_val2[mask == 1], corr_test2[mask == 1]
    new_nse_tr2, new_nse_val2, new_nse_test2 = nse_tr2[mask == 1], nse_val2[mask == 1], nse_test2[mask == 1]
    new_nrmse_tr2, new_nrmse_val2, new_nrmse_test2 = nrmse_tr2[mask == 1], nrmse_val2[mask == 1], nrmse_test2[mask == 1]

    DA_UNet = io.loadmat('perf_withDA_noise2.mat')
    corr_tr3, corr_val3, corr_test3 = DA_UNet['corr_tr'], DA_UNet['corr_val'], DA_UNet['corr_test']
    nse_tr3, nse_val3, nse_test3 = DA_UNet['nse_tr'], DA_UNet['nse_val'], DA_UNet['nse_test']
    nrmse_tr3, nrmse_val3, nrmse_test3 = DA_UNet['nrmse_tr'], DA_UNet['nrmse_val'], DA_UNet['nrmse_test']
    new_corr_tr3, new_corr_val3, new_corr_test3 = corr_tr3[mask == 1], corr_val3[mask == 1], corr_test3[mask == 1]
    new_nse_tr3, new_nse_val3, new_nse_test3 = nse_tr3[mask == 1], nse_val3[mask == 1], nse_test3[mask == 1]
    new_nrmse_tr3, new_nrmse_val3, new_nrmse_test3 = nrmse_tr3[mask == 1], nrmse_val3[mask == 1], nrmse_test3[mask == 1]

    # prepare scatter plot data
    gldas = np.load('gldas_data.npz', allow_pickle=True)
    g_grid = gldas['g_grid'][27:, ]  # from 2002-4 to 2021-11
    # remove the gldas data during the data gap period
    g_grid = np.concatenate((g_grid[0:183, ...], g_grid[194:236, ...]), axis=0)
    g_grid = g_grid[:, 103:135, 163:243]

    # 2 GRACE
    grace = np.load('targets.npz', allow_pickle=True)
    grace_grid = grace['grace']  # 183*32*80  S*H*W
    gfo_grid = grace['gfo']  # 42*32*80
    G_grid = np.concatenate((grace_grid, gfo_grid), axis=0)

    from networks import U_Net
    import torch
    model_withoutDA = U_Net()
    model_withoutDA.to('cpu')
    model_withoutDA.eval()
    model_withoutDA.load_state_dict(torch.load('model_withoutDA.pt'))

    model_DA = U_Net()
    model_DA.to('cpu')
    model_DA.eval()
    model_DA.load_state_dict(torch.load('model_withDA_noise2.pt'))

    x = np.load('inputs.npz')
    grid1 = x['g1']  # 183*3*32*80   S * C * H * W = S * 3 * 32 * 80  (S for samples)
    grid2 = x['g2']  # 42*3*32*80
    inputs_grid = np.concatenate((grid1, grid2), axis=0)  # 225*3*32*80
    dtype = torch.float32
    input_t = torch.from_numpy(inputs_grid).to(dtype=dtype)

    out_withoutDA = model_withoutDA(input_t)
    out_DA = model_DA(input_t)

    out_withoutDA_a = out_withoutDA.detach().squeeze().numpy()
    out_DA_a = out_DA.detach().squeeze().numpy()

    mask_new = np.load('mask.npz', allow_pickle=True)['mask']
    mask_new[mask_new == 0] = np.nan

    flag = UNet['basin_result'][:, 3]  # 1, 2,3 represent training, validating and testing, respectively
    if period == 'test':
        period_flag = 3
    elif period == 'val':
        period_flag = 2
    else:
        period_flag = 1

    gldas_grid = (g_grid[flag == period_flag] * mask_new).flatten()
    grace_gfo_grid = (G_grid[flag == period_flag] * mask_new).flatten()
    unet_grid = (out_withoutDA_a[flag == period_flag] * mask_new).flatten()
    na_unet_grid = (out_DA_a[flag == period_flag] * mask_new).flatten()

    grid_list = list([gldas_grid, unet_grid, na_unet_grid])
    # ----------------------------------------------------------------------------------------------------------

    if period == 'test':
        new_corr_test1 = np.sort(new_corr_test1)
        new_corr_test2 = np.sort(new_corr_test2)
        new_corr_test3 = np.sort(new_corr_test3)
        new_corr_test = [new_corr_test1, new_corr_test2, new_corr_test3]
        data1 = new_corr_test

        new_nse_test1 = np.sort(new_nse_test1[new_nse_test1 >= -1])
        new_nse_test2 = np.sort(new_nse_test2[new_nse_test2 >= -1])
        new_nse_test3 = np.sort(new_nse_test3[new_nse_test3 >= -1])
        new_nse_test = [new_nse_test1, new_nse_test2, new_nse_test3]
        data2 = new_nse_test

        new_nrmse_test1 = np.sort(new_nrmse_test1[new_nrmse_test1 <= 1])
        new_nrmse_test2 = np.sort(new_nrmse_test2[new_nrmse_test2 <= 1])
        new_nrmse_test3 = np.sort(new_nrmse_test3[new_nrmse_test3 <= 1])
        new_nrmse_test = [new_nrmse_test1, new_nrmse_test2, new_nrmse_test3]
        data3 = new_nrmse_test
    elif period == 'val':
        new_corr_val1 = np.sort(new_corr_val1)
        new_corr_val2 = np.sort(new_corr_val2)
        new_corr_val3 = np.sort(new_corr_val3)
        new_corr_val = [new_corr_val1, new_corr_val2, new_corr_val3]
        data1 = new_corr_val

        new_nse_val1 = np.sort(new_nse_val1[new_nse_val1 >= -1])
        new_nse_val2 = np.sort(new_nse_val2[new_nse_val2 >= -1])
        new_nse_val3 = np.sort(new_nse_val3[new_nse_val3 >= -1])
        new_nse_val = [new_nse_val1, new_nse_val2, new_nse_val3]
        data2 = new_nse_val

        new_nrmse_val1 = np.sort(new_nrmse_val1[new_nrmse_val1 <= 2])
        new_nrmse_val2 = np.sort(new_nrmse_val2[new_nrmse_val2 <= 2])
        new_nrmse_val3 = np.sort(new_nrmse_val3[new_nrmse_val3 <= 2])
        new_nrmse_val = [new_nrmse_val1, new_nrmse_val2, new_nrmse_val3]
        data3 = new_nrmse_val
    else:
        new_corr_tr1 = np.sort(new_corr_tr1)
        new_corr_tr2 = np.sort(new_corr_tr2)
        new_corr_tr3 = np.sort(new_corr_tr3)
        new_corr_tr = [new_corr_tr1, new_corr_tr2, new_corr_tr3]
        data1 = new_corr_tr

        new_nse_tr1 = np.sort(new_nse_tr1[new_nse_tr1 >= -1])
        new_nse_tr2 = np.sort(new_nse_tr2[new_nse_tr2 >= -1])
        new_nse_tr3 = np.sort(new_nse_tr3[new_nse_tr3 >= -1])
        new_nse_tr = [new_nse_tr1, new_nse_tr2, new_nse_tr3]
        data2 = new_nse_tr

        new_nrmse_tr1 = np.sort(new_nrmse_tr1[new_nrmse_tr1 <= 2])
        new_nrmse_tr2 = np.sort(new_nrmse_tr2[new_nrmse_tr2 <= 2])
        new_nrmse_tr3 = np.sort(new_nrmse_tr3[new_nrmse_tr3 <= 2])
        new_nrmse_tr = [new_nrmse_tr1, new_nrmse_tr2, new_nrmse_tr3]
        data3 = new_nrmse_tr

    data = list([data1, data2, data3])
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    box_label = ['CC', 'NSE', 'NRMSE']
    label_box = 'abc'

    for idx, ax in enumerate(axes[0]):
        ax.boxplot(data[idx][0], patch_artist=True, positions=[1], labels=['GLDAS'],
                   showfliers=False, boxprops={"facecolor": "#614ad3", "edgecolor": "none", 'alpha': 0.5}, widths=0.4)

        ax.boxplot(data[idx][1], patch_artist=True, positions=[2], labels=['UNet'],
                   showfliers=False, boxprops={"facecolor": "#614ad3", "edgecolor": "none", 'alpha': 0.5}, widths=0.4)

        ax.yaxis.grid(True)
        ax.spines['left'].set_color('#614ad3')
        ax.tick_params(axis='y', colors='#614ad3')
        ax.set_title(box_label[idx])
        ax.text(s=label_box[idx], fontweight='bold', x=0.06, y=0.94,
                transform=ax.transAxes, horizontalalignment='right', fontsize=10)

        ax_NA = ax.twinx()
        ax_NA.boxplot(data[idx][2], patch_artist=True, positions=[3], labels=['NA-UNet'],
                      showfliers=False, boxprops={"facecolor": '#e42c64', "edgecolor": "none", 'alpha': 0.5},
                      widths=0.4)
        ax_NA.spines['right'].set_color('red')
        ax_NA.tick_params(axis='y', colors='red')

    scatter_label = list(['GLDAS', 'UNet', 'NA-UNet'])
    label_scatter = 'def'
    for idx, ax in enumerate(axes[1]):
        ax.scatter(grace_gfo_grid, grid_list[idx], c='#F78500', alpha=0.5, s=7)
        ax.set_ylabel('GRACE TWSA [cm]')
        ax.set_xlabel(f'{scatter_label[idx]} TWSA [cm]')

        ax.plot([0, 1], [0, 1], transform=ax.transAxes, c='k', ls='--')
        ax.text(s=label_scatter[idx], fontweight='bold', x=0.06, y=0.92,
                transform=ax.transAxes, horizontalalignment='right', fontsize=10)

        ax.yaxis.grid(True)

    fig.subplots_adjust(wspace=0.5, left=0.08, right=0.95, hspace=0.4)

    if save_flag:
        plt.savefig('python_figs/Fig6.jpg', dpi=800)

    if show_flag:
        plt.show()


def gap_simulate(model_pt='sim_model.pt', show_flag=True):
    from networks import U_Net
    import torch
    model = U_Net()
    model.to('cpu')
    model.eval()
    model.load_state_dict(torch.load(model_pt))

    input_data = np.load('inputs.npz')
    target_data = np.load('targets.npz')

    x = input_data['g1']
    y = target_data['grace']

    gap_input, gap_target = x[-12:], y[-12:]  # 11*3*32*80   11*32*80 2016-07~2017-06

    dtype = torch.float32
    input_t = torch.from_numpy(gap_input).to(dtype=dtype)

    model_outputs = model(input_t)
    model_a = model_outputs.detach().squeeze().numpy()

    # pre, obs = model_a, gap_target
    # mask = np.load('mask.npz', allow_pickle=True)['mask']
    # weight = np.load('grid_weight.npz', allow_pickle=True)['grid_weight']
    #
    # p_s = np.array([np.nansum(grid * weight * mask) / np.nansum(mask * weight) for grid in pre])
    # o_s = np.array([np.nansum(grid * weight * mask) / np.nansum(mask * weight) for grid in obs])
    #
    # plt.plot(np.arange(len(p_s)), p_s, o_s)
    #
    # if show_flag:
    #     plt.show()

    return model_a, gap_target


def plot_sim_spatial(show_flag=True, save_flag=False):
    plt.style.use('seaborn')
    sim_data = np.load('sim_pre_obs.npz', allow_pickle=True)
    pre, obs = sim_data['pre'], sim_data['obs']

    dates_list = [['2016-07', '2016-08', '2016-09'], ['2016-10', '2016-11', '2016-12'],
                  ['2017-01', '2017-02', '2017-03'], ['2017-04', '2017-05', '2017-06']]

    fig = plt.figure(layout="constrained", figsize=(13, 7))
    subfigs = fig.subfigures(1, 2, wspace=0)
    axs0 = subfigs[0].subplots(4, 3)

    # subfigs[0].set_facecolor('#08ffc8')
    # subfigs[0].patch.set_alpha(0.2)
    # subfigs[1].set_facecolor('#5BE7C4')
    # subfigs[1].patch.set_alpha(0.2)

    mask = np.load('mask.npz', allow_pickle=True)['mask']
    mask[mask == 0] = np.nan
    cmap = 'gist_rainbow'
    for idx, axes in enumerate(axs0):
        for jdx, ax in enumerate(axes):
            img = ax.contourf(np.flipud(pre[idx] * mask), cmap=cmap, levels=25)
            ax.axis('off')
            ax.set_title(dates_list[idx][jdx])

    axs0[0, 0].text(s='a', fontweight='bold', x=0.04, y=0.98,
                    transform=axs0[0, 0].transAxes, horizontalalignment='right', fontsize=14)

    ticks = np.arange(-20, 26, 5)
    cbar = fig.colorbar(img, ax=axs0, shrink=0.5, pad=0.03, orientation='horizontal', ticks=ticks, )
    cbar.set_label(label='Predicted TWSA [cm]', size=12, weight='bold')

    axs1 = subfigs[1].subplots(4, 3)
    for idx, axes in enumerate(axs1):
        for jdx, ax in enumerate(axes):
            img1 = ax.contourf(np.flipud(obs[idx] * mask), cmap=cmap, levels=25)
            ax.axis('off')
            ax.set_title(dates_list[idx][jdx])

    cbar1 = fig.colorbar(img1, ax=axs1, shrink=0.5, pad=0.03, orientation='horizontal', ticks=ticks)
    cbar1.set_label(label='Observed TWSA [cm]', size=12, weight='bold')

    cbar1.ax.tick_params(labelsize=12)
    cbar.ax.tick_params(labelsize=12)

    axs1[0, 0].text(s='b', fontweight='bold', x=0.04, y=0.98,
                    transform=axs1[0, 0].transAxes, horizontalalignment='right', fontsize=14)

    if save_flag:
        plt.savefig('python_figs/Fig7.jpg', dpi=1000)

    if show_flag:
        plt.show()


if __name__ == '__main__':
    print('start...')
    plot_loss(save_flag=True)
    # pre, obs = gap_simulate(show_flag=False)
    # # mask = np.load('mask.npz', allow_pickle=True)['mask']
    # # weight = np.load('grid_weight.npz', allow_pickle=True)['grid_weight']
    # #
    # # p_s = np.array([np.nansum(grid * weight * mask) / np.nansum(mask * weight) for grid in pre])
    # # o_s = np.array([np.nansum(grid * weight * mask) / np.nansum(mask * weight) for grid in obs])
    # #
    # # plt.plot(np.arange(len(p_s)), p_s, o_s)
    # # plt.show()
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(pre[0])
    # ax[1].imshow(obs[0])
    # plt.show()
