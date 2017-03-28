from planet4 import io, markings, region_data
import numpy as np
from little_helpers.p4_tools import get_final_markings_counts
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from little_helpers.spherical_trigonometry import arc_length, triangle_area
import os


meta_data_fn = Path.home() / 'Dropbox/SternchenAndMe/python_stuff/hirise_rdr_index.hdf'


def get_metadata(region, season=None):
    if season is not None:
        fname = f"{region}_season{season}_metadata.csv"
        df = pd.read_csv(io.analysis_folder() / fname)
        df = df.drop('path', axis=1)
    else:
        fglob = f"{region}_season*_metadata.csv"
        globpath = io.analysis_folder() / fglob
        bucket = []
        for fname in globpath:
            bucket.append(pd.read_csv(fname))
        df = pd.concat(bucket, ignore_index=False)
        df = df.drop('path', axis=1)
    return df


def scaling_blotch(obsid, BlAreaCut=False, catalog_name='p4_catalog'):
    catalog_folder = io.analysis_folder() / catalog_name
    bl_file2open = catalog_folder / (obsid + '_blotches.csv')
    bc = markings.BlotchContainer.from_fname(bl_file2open)
    if BlAreaCut:
        all_bl_areas = np.array([obj.area for obj in bc.content])
        min_bl_area = Inca.at[obsid, 'min_bl_area']
        # TODO: Ask Anya about this never being used?
        nr_bl2subtract = len(all_bl_areas[all_bl_areas > min_bl_area])
    else:
        all_bl_rad1 = np.array([obj.data.radius_1 for obj in bc.content])
        # maybe make min of these?
        all_bl_rad2 = np.array([obj.data.radius_2 for obj in bc.content])
        min_bl = Inca.at[obsid, 'min_bl_radius']
        nr_bl_red = len(all_bl_rad1[all_bl_rad1 > min_bl])
    return nr_bl_red


def scaling_fan(obsid):
    catalog_folder = io.analysis_folder() / 'p4_catalog'
    fan_file2open = catalog_folder / (obsid + '_fans.csv')
    fc = markings.FanContainer.from_fname(fan_file2open)
    all_fan_length = np.array([obj.data.distance for obj in fc.content])
    min_fan = Inca.at[obsid, 'min_fan_length']
    nr_fans_red = len(all_fan_length[all_fan_length > min_fan])
    return nr_fans_red
