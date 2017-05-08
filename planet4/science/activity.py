import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import arccos, cos, sin, pi

# from little_helpers.p4_tools import get_final_markings_counts
from planet4 import io, markings, region_data

meta_data_fn = Path.home() / 'Dropbox/SternchenAndMe/python_stuff/hirise_rdr_index.hdf'


def arc_length(lat1, lon1, lat2, lon2):
    """calculate length of arc from coordinates of end points

    works for east longitudes and south latitdes (with +)
    """
    phi1 = pi + lat1
    phi2 = pi + lat2

    AB = sin(phi1) * sin(phi2) + cos(-lon1 + lon2) * cos(phi1) * cos(phi2)
    arc = arccos(AB)
    return arc


def spherical_excess(a, b, c):
    "spherical excess of the triangle."
    A = arccos((cos(a) - cos(b) * cos(c)) / sin(b) / sin(c))
    B = arccos((cos(b) - cos(c) * cos(a)) / sin(c) / sin(a))
    C = arccos((cos(c) - cos(a) * cos(b)) / sin(a) / sin(b))
    E = A + B + C - pi
    return(E)


def triangle_area(a, b, c):
    '''calculate area of a spherical triangle'''
    R_Mars = 3376.20000  # in km
    E = spherical_excess(a, b, c)
    Area = R_Mars**2 * E
    return Area


def get_metadata(region, season=None):
    if season is not None:
        fname = f"{region}_season{season}_metadata.csv"
        df = pd.read_csv(io.analysis_folder() / fname)
        df = df.drop('path', axis=1)
    else:
        fglob = f"{region}_season*_metadata.csv"
        globpath = io.analysis_folder().glob(fglob)
        bucket = []
        for fname in globpath:
            bucket.append(pd.read_csv(fname))
        df = pd.concat(bucket, ignore_index=False)
        df = df.drop('path', axis=1)
    return df


def get_final_markings_counts(root, img_name, cut=0.5):
    # writing in dictionary here b/c later I convert it to pd.DataFrame
    # for which a dictionary is a natural input format
    d = {}
    d['obsid'] = img_name
    blotch_fname = root / '{}_blotches.csv'.format(img_name)  # was _latlon
    d['n_blotches'] = len(pd.read_csv(str(blotch_fname)))
    fan_fname = root / '{}_fans.csv'.format(img_name)  # was '{}_fans_latlons.csv'.format(img_name)
    d['n_fans'] = len(pd.read_csv(str(fan_fname)))

    return d


def scaling_blotch(row, BlAreaCut=False, catalog_name='p4_catalog'):
    obsid = row.obsid
    catalog_folder = io.analysis_folder() / catalog_name
    bl_file2open = catalog_folder / (obsid + '_blotches.csv')
    bc = markings.BlotchContainer.from_fname(bl_file2open)
    if BlAreaCut:
        all_bl_areas = np.array([obj.area for obj in bc.content])
        min_bl_area = row.min_bl_area
        # TODO: Ask Anya about this never being used?
        nr_bl2subtract = len(all_bl_areas[all_bl_areas > min_bl_area])
    else:
        all_bl_rad1 = np.array([obj.data.radius_1 for obj in bc.content])
        # maybe make min of these?
        all_bl_rad2 = np.array([obj.data.radius_2 for obj in bc.content])
        min_bl = row.min_bl_radius
        nr_bl_red = len(all_bl_rad1[all_bl_rad1 > min_bl])
    return nr_bl_red


def scaling_fan(row):
    obsid = row.obsid
    min_fan = row.min_fan_length
    catalog_folder = io.analysis_folder() / 'p4_catalog'
    fan_file2open = catalog_folder / (obsid + '_fans.csv')
    fc = markings.FanContainer.from_fname(fan_file2open)
    all_fan_length = np.array([obj.data.distance for obj in fc.content])
    nr_fans_red = len(all_fan_length[all_fan_length > min_fan])
    return nr_fans_red
