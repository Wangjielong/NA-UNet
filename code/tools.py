from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import math


def get_date(days, start_date=datetime(year=2002, month=1, day=1)):
    """
    According to the days since the start_date to calculate the current datetime
    :param days: the days since the start_date
    :param start_date: default 2002-1-1
    :return: current date (type: datetime)
    """
    duration = timedelta(days=days)
    current_date = start_date + duration
    return current_date


def region_mask(boundary_file='yangtze.txt', resolution=0.5):
    """
    according to the boundary file to calculate the region mask
    :param resolution: the spatial resolution of mask we create
                       default: 0.5 degree
    :param boundary_file: the filename of specified region boundary
                default: yangtze.txt
    :return: mask: the mask of interested region or basin (type: ndarray)
    """
    # boundary file should be *.txt
    # Read the boundary file
    boundary = pd.read_csv(boundary_file, header=None, sep=r"\s+", names=['lon', 'lat'])
    region_polygon = Polygon([(lon, lat) for lon, lat in boundary.values])
    lon_min = min(boundary['lon'])
    lon_max = max(boundary['lon'])
    lat_min = min(boundary['lat'])
    lat_max = max(boundary['lat'])

    jj = int(180 / resolution)  # the dimension of latitude
    ii = int(360 / resolution)  # the dimension of longitude
    # longitude from left to right is 0.25°E to 359.75°E instead of 180°W to 180°E and
    # latitude from top to bottom is North to South
    # np.newaxis exists for we don't want to lose dimension information
    lon = np.arange(start=0.5 * resolution, stop=360, step=resolution)[:, np.newaxis]
    lat = np.arange(start=90 - 0.5 * resolution, stop=-90, step=-resolution)[:, np.newaxis]
    mask = np.zeros((jj, ii))
    for j in range(jj):
        for i in range(ii):
            if lon_min <= lon[i] <= lon_max and lat_min <= lat[j] <= lat_max:
                point = Point(lon[i], lat[j])
                if region_polygon.intersects(point):
                    mask[j][i] = 1
    return mask


def lat_weighted(boundary_file='yangtze.txt', resolution=0.5):
    """
    according to the region boundary to calculate the grid weight and total weight
    :param resolution: the spatial resolution of mask we create
                       default: 0.5 degree
    :param boundary_file: the filename of specified region boundary
                       default: yangtze.txt
    :return: grid weight (ndarray) and total weight
    """
    mask = region_mask(boundary_file, resolution)
    jj = int(180 / resolution)  # the dimension of latitude
    ii = int(360 / resolution)  # the dimension of longitude
    #  latitude from top to bottom is North to South
    lat = np.arange(start=90 - 0.5 * resolution, stop=-90, step=-resolution)[:, np.newaxis]
    grid_weight = np.zeros((jj, ii))
    for j in range(jj):
        for i in range(ii):
            grid_weight[j][i] = math.cos(lat[j] * math.pi / 180)  # the cosine of latitude as weight

    grid_weight = grid_weight * mask
    total_weight = np.sum(grid_weight)
    return grid_weight, total_weight


def lat_lon_gen(resolution=0.5):
    """
    generate the (lat, lon) grid of resolution; the resulting grid has two channels
    the first chanel consists of the lat(s) while the second chanel consists of the lon(s)
    :param resolution: the grid resolution; default=0.5 deg; 2*360*720
    :return: a lat-lon grid for the world
    """
    ii = int(180 / resolution)  # the dimension of latitude
    jj = int(360 / resolution)  # the dimension of longitude
    grid = np.empty((2, ii, jj))
    start_lat = 90 - 0.5 * resolution
    start_lon = 0.5 * resolution

    for i in range(ii):
        grid[0, i, :] = start_lat - i * resolution
    for j in range(jj):
        grid[1, :, j] = start_lon + j * resolution

    return grid


grid1 = lat_lon_gen()
lat = grid1[0, 103:135, 163: 243]
lon= grid1[1, 103:135, 163:243]
