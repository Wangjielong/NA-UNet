import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.shapereader as sr
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import numpy as np
import matplotlib as mpl
import geopandas as gpd
import xarray as xr

plt.rc('font', family='sans-serif')

# # 读取全球地形数据
ds = xr.open_dataset(r'D:\\Doctoral Documents\\program\\data\\Elevation\\ETOPO2v2c_f4.nc')
# 准备用于绘图的数据
lon = np.linspace(min(ds['x'].data), max(ds['x'].data), len(ds['x'].data))  # 经度
lat = np.linspace(min(ds['y'].data), max(ds['y'].data), len(ds['y'].data))  # 纬度
lon, lat = np.meshgrid(lon, lat)  # 构建经纬网
dem = ds['z'].data  # DEM数据


def plot_yangtze(show_flag=True, save_flag=False):
    params = {
        'axes.linewidth': 2,
        'figure.facecolor': 'w',
    }
    mpl.rcParams.update(params)

    proj = ccrs.PlateCarree()
    extent = [85, 123, 20, 40]

    fig = plt.figure(figsize=(16, 10), layout='constrained')
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(extent, crs=ccrs.PlateCarree())  # set the plotting extent

    # adding the DEM
    levels = np.arange(0, 8001, 500)
    cf = ax.contourf(lon, lat, dem, levels=levels, cmap='RdPu')
    position = fig.add_axes([0.65, 0.78, 0.03, 0.08])  # left bottom width height
    cb = fig.colorbar(cf, cax=position, orientation='vertical', ticks=[0, 8000])
    cb.ax.set_title('DEM (m)')
    cb.ax.tick_params(labelsize=10)

    up_yrb_path = r'D:\Doctoral Documents\program\data\Yangtze\Yangtze from Di\UPYangtze.shp'
    down_yrb_path = r'D:\Doctoral Documents\program\data\Yangtze\Yangtze from Di\DOWNYangtze.shp'
    river_path = r'D:\Doctoral Documents\program\data\Yangtze\Changjiang\rivers.shp'
    river2_path = r'D:\Doctoral Documents\program\data\Yangtze\yangtze vector\河流1-3级.shp'

    up_yrb_mask = sr.Reader(up_yrb_path)
    down_yrb_mask = sr.Reader(down_yrb_path)
    river = sr.Reader(river_path)
    river2 = sr.Reader(river2_path)

    ax.add_geometries(up_yrb_mask.geometries(), crs=proj, linewidths=2, edgecolor='k',
                      facecolor='none', alpha=0.5)  # add the region mask
    ax.add_geometries(down_yrb_mask.geometries(), crs=proj, linewidths=1.5, edgecolor='k',
                      facecolor='none', alpha=0.5)  # add the region mask
    ax.add_geometries(river.geometries(), crs=proj, linewidths=2, edgecolor='lightblue',
                      facecolor='none', zorder=5, alpha=1)  # add the river
    ax.add_geometries(river2.geometries(), crs=proj, linewidths=1, edgecolor='lightblue',
                      facecolor='none', zorder=5, )  # add the river

    # adding the hydrological station
    ax.annotate(text='Dongting', size=15, xy=(111.5, 28.1588), xytext=(111.5, 28.1588),
                color='b', zorder=7)
    ax.annotate(text='Poyang', size=15, xy=(115.5, 28), xytext=(115.5, 28),
                color='b', zorder=7)
    ax.annotate(text='Taihu', size=15, xy=(118.5, 31), xytext=(118.5, 31),
                color='b', zorder=7)

    import pandas as pd
    pd_city = pd.read_excel('city.xlsx')
    for row in pd_city.itertuples():
        ax.annotate(text=row[-1], size=12, xy=(row[1], row[2]), xytext=(row[1], row[2]),
                    color='k', zorder=7)

    ax.plot(110.9, 31, marker='p', c='gray', markersize=12, zorder=7, label='Three Gorges Dam')  # TGD
    ax.annotate(text='TGD', size=15, xy=(110.9, 31), xytext=(110.5, 31.4), fontweight='bold', color='k', zorder=7)

    # adding the main stream
    ms_shp = r'D:\Doctoral Documents\program\data\Yangtze\Changjiang\rivers.shp'
    gdf = gpd.read_file(ms_shp)
    ms_geo = gdf['geometry'][93]
    ax.add_geometries([ms_geo], crs=proj, linewidths=2, edgecolor='b',
                      facecolor='none', zorder=6, alpha=0.8)  # add the main stream
    ax.plot([91, 91], [22, 22], lw=2, c='b', label='Yangtze River')
    ax.plot([91, 91], [22, 22], lw=2, c='lightblue', label='Stream')

    # adding legend
    ax.legend(frameon=True, bbox_to_anchor=(0.65, 0.98), prop={'size': 15}, )

    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))  # add the longitude
    ax.yaxis.set_major_formatter(LatitudeFormatter())  # add the latitude
    ax.set_xticks(np.arange(extent[0], extent[1] + 1, 4), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(extent[2] + 2, extent[3] + 1, 3), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN)  # add the ocean
    ax.add_feature(cfeature.LAND.with_scale('10m'), alpha=0.75)  # add the land
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), lw=0.5)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), color='lightblue')
    ax.tick_params(labelsize=12)
    ax.tick_params(right=True, top=True, labelright=True, labeltop=True, labelrotation=0)
    ax.grid(linestyle='--', color='gray', alpha=0.5)

    # adding the inset
    # crs = ccrs.Orthographic(90, 30)
    ax_in = plt.axes((0.033, 0.110, 0.20, 0.20), projection=proj)
    # ax_in.add_feature(cfeature.OCEAN, zorder=0)
    # ax_in.add_feature(cfeature.LAND, zorder=0, color='w')
    ax_in.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--')
    ax_in.set_extent([72, 136, 14, 55], crs=ccrs.PlateCarree())  # set the plotting extent

    china_path = r'D:\Doctoral Documents\program\data\Yangtze\vector data\1国界\China_World Mercator.shp'
    china_mask = sr.Reader(china_path)

    ax_in.add_geometries(up_yrb_mask.geometries(), crs=ccrs.PlateCarree(), linewidths=0.5, edgecolor='none',
                         facecolor='r', zorder=1, alpha=0.8)  # add the region mask
    ax_in.add_geometries(down_yrb_mask.geometries(), crs=proj, linewidths=0.5, edgecolor='none',
                         facecolor='r', zorder=1, alpha=0.8)  # add the region mask
    ax_in.add_geometries(china_mask.geometries(), crs=ccrs.Mercator(), linewidths=1, edgecolor='k',
                         facecolor='none', zorder=2)  # add the region mask

    ax_in.text(s='Chinese Mainland', fontweight='bold', x=0.70, y=0.55,
               transform=ax_in.transAxes, horizontalalignment='right', fontsize=12)

    ax_in.text(s='YRB', fontweight='bold', x=0.63, y=0.36,
               transform=ax_in.transAxes, horizontalalignment='right', fontsize=12)

    # adding the UYRB and LYRB
    # arrowprops = dict(arrowstyle="-", color='k')
    # ax.annotate(text='UYRB', size=15, xy=(102, 29), xytext=(96, 26), arrowprops=arrowprops, zorder=11)
    # ax.annotate(text='LYRB', size=15, xy=(112, 27), xytext=(110, 23), arrowprops=arrowprops, zorder=11)

    if save_flag:
        plt.savefig('python_figs/Fig1.jpg', dpi=800)

    if show_flag:
        plt.show()


if __name__ == '__main__':
    print('starting...')
    plot_yangtze(save_flag=True)
    plt.show()
