"""This module contains code for binding together the clustering and fnotching tools to create the
final catalog files.
When run as a script from command-line, it requires to launch a local ipcontroller for the parallel
processing.
If you execute this locally, you can create one with `ipcluster start -n <no>`, with <no> the number
of cores you want to provide to the parallel processing routines.
"""
import argparse
import logging

import pandas as pd
from ipyparallel import Client
from ipyparallel.util import interactive
from tqdm import tqdm

from nbtools import execute_in_parallel

from . import io
from .metadata import MetadataReader
from .projection import create_RED45_mosaic, TileCalculator, xy_to_hirise


LOGGER = logging.getLogger(__name__)
# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def cluster_obsid(obsid=None, savedir=None, fnotch_via_obsid=False,
                  imgid=None):
    """Cluster all image_ids for given obsid (=image_name).

    Parameters
    ----------
    obsid : str
        HiRISE obsid (= Planet four image_name)
    savedir : str or pathlib.Path
        Top directory path where the catalog will be stored. Will create folder if it
        does not exist yet.
    fnotch_via_obsid : bool, optional
        Switch to control if fnotching happens per image_id or per obsid
    imgid : str, optional
        Convenience parameter: If `obsid` is not given, and `obsid` is none, this `image_id` can
        be used to receive the respective `obsid` from the ImageID class.
    """
    from planet4 import dbscan, fnotching, markings

    # parameter checks
    if obsid is None and imgid is not None:
        obsid = markings.ImageID(imgid).image_name
    elif obsid is None and imgid is None:
        raise ValueError("Provide either obsid or imgid.")

    # cluster
    dbscanner = dbscan.DBScanner(savedir=savedir)
    dbscanner.cluster_image_name(obsid)

    # fnotching / combining ambiguous cluster results
    # fnotch across all the HiRISE image
    # does not work yet correctly! Needs to scale for n_classifications
    if fnotch_via_obsid is True:
        fnotching.fnotch_obsid(obsid, savedir=savedir)
        fnotching.apply_cut_obsid(obsid, savedir=savedir)
    else:
        # default case: Fnotch for each image_id separately.
        fnotching.fnotch_image_ids(obsid, savedir=savedir)
        fnotching.apply_cut(obsid, savedir=savedir)
    return obsid


def process_obsid_parallel(args):
    obsid, savedir = args
    return cluster_obsid(obsid, savedir)


class ReleaseManager:
    def __init__(self, version, obsids=None, overwrite=False):
        self.catalog = f'P4_catalog_{version}'
        self.overwrite = overwrite
        self._obsids = obsids

    @property
    def savefolder(self):
        return io.data_root / self.catalog

    @property
    def metadata_path(self):
        return self.savefolder / f"{self.catalog}_metadata.csv"

    @property
    def tile_coords_path(self):
        return self.savefolder / f"{self.catalog}_tile_coords.csv"

    @property
    def obsids(self):
        if self._obsids is None:
            db = io.DBManager()
            self._obsids = db.obsids
        return self._obsids

    @obsids.setter
    def obsids(self, values):
        self._obsids = values

    @property
    def fan_file(self):
        return next(self.savefolder.glob("*_fan.csv"))

    @property
    def blotch_file(self):
        return next(self.savefolder.glob("*_blotch.csv"))

    def create_parallel_args(self):
        bucket = []
        for obsid in self.obsids:
            pm = io.PathManager(obsid=obsid, datapath=savedir)
            path = pm.obsid_results_savefolder / obsid
            if path.exists() and self.overwrite is False:
                continue
            else:
                bucket.append(obsid)
        self.todo = [(i, self.catalog) for i in bucket]

    def get_metadata(self):
        metadata = []
        for img in tqdm(self.obsids):
            metadata.append(MetadataReader(img).get_data_dic())
        df = pd.DataFrame(metadata)
        df.to_csv(self.metadata_path, index=False)

    def get_tile_coordinates(self):
        edrpath = io.get_ground_projection_root()
        cubepaths = [edrpath / obsid / f"{obsid}_mosaic_RED45.cub" for obsid in obsids]
        todo = []
        for cubepath in cubepaths:
            tc = TileCalculator(cubepath, read_data=False)
            if not tc.campt_results_path.exists():
                todo.append(cubepath)

        def get_tile_coords(cubepath):
            from planet4.projection import TileCalculator
            tilecalc = TileCalculator(cubepath)
            tilecalc.calc_tile_coords()
        results = execute_in_parallel(get_tile_coords, todo)

        bucket = []
        for cubepath in tqdm(cubepaths):
            tc = TileCalculator(cubepath, read_data=False)
            bucket.append(tc.tile_coords_df)
        coords = pd.concat(bucket, ignore_index=True)
        coords.to_csv(self.tile_coords_path, index=False)

    def launch_catalog_production(self):
        # perform the clustering
        LOGGER.info("Performing the clustering.")
        results = execute_in_parallel(process_obsid_parallel, self.todo)
        # create summary CSV files of the clustering output
        LOGGER.info("Creating L1C fan and blotch database files.")
        create_roi_file(self.obsids, self.catalog, self.catalog)
        # create the RED45 mosaics for all ground_projection calculations
        LOGGER.info("Creating the required RED45 mosaics for ground projections.")
        results = execute_in_parallel(create_RED45_mosaic, self.obsids)
        # calculate center ground coordinates for all tiles involved
        LOGGER.info("Calculating the ground coordinates for all P4 tiles.")
        self.get_tile_coordinates()
        # calculate all metadata required for P4 analysis
        LOGGER.info("Writing summary metadata file.")
        self.get_metadata()

            


@interactive
def do_clustering(p4img, kind='fans'):
    from planet4 import clustering
    import pandas as pd

    reduced = clustering.perform_dbscan(p4img, kind)
    if reduced is None:
        return None
    series = [cluster.data for cluster in reduced]
    n_members = [cluster.n_members for cluster in reduced]
    n_rejected = [cluster.n_rejected for cluster in reduced]
    df = pd.DataFrame(series)
    df['image_id'] = p4img.imgid
    df['n_members'] = n_members
    df['n_rejected'] = n_rejected
    return df


@interactive
def process_image_name(image_name):
    from os.path import join as pjoin
    import os
    import pandas as pd
    from planet4 import markings
    HOME = os.environ['HOME']

    dirname = pjoin(HOME, 'data/planet4/catalog_2_and_3')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    blotchfname = pjoin(dirname, image_name + '_reduced_blotches.hdf')
    fanfname = pjoin(dirname, image_name + '_reduced_fans.hdf')
    if os.path.exists(blotchfname) and\
            os.path.exists(fanfname):
        return image_name + ' already done.'
    db = io.DBManager()
    data = db.get_image_name_markings(image_name)
    img_ids = data.image_id.unique()
    blotches = []
    fans = []
    for img_id in img_ids:
        p4img = markings.ImageID(img_id)
        blotches.append(do_clustering(p4img, 'blotches'))
        fans.append(do_clustering(p4img, 'fans'))
    blotches = pd.concat(blotches, ignore_index=True)
    blotches.to_hdf(blotchfname, 'df')
    fans = pd.concat(fans, ignore_index=True)
    fans.to_hdf(fanfname, 'df')
    return image_name


def read_csvfiles_into_lists_of_frames(folders):
    bucket = dict(fan=[], blotch=[])
    for folder in folders:
        for markingfile in folder.glob("*.csv"):
            key = 'fan' if markingfile.name.endswith('fans.csv') else 'blotch'
            bucket[key].append(pd.read_csv(markingfile))
    return bucket


def create_roi_file(obsids, roi_name, datapath):
    """Create a Region of Interest file, based on list of obsids.

    For more structured analysis processes, we can create a summary file for a list of obsids
    belonging to a ROI.
    The alternative is to define to what ROI any final object belongs to and add that as a column
    in the final catalog.

    Parameters
    ----------
    obsids : iterable of str
        List of HiRISE obsids
    roi_name : str
        Name for ROI
    datapath : str or pathlib.Path
        Path to the top folder with the clustering output data.
    """
    Bucket = dict(fan=[], blotch=[])
    for obsid in tqdm(obsids):
        pm = io.PathManager(obsid=obsid, datapath=datapath)
        # get all L1C folders for current obsid:
        folders = pm.get_obsid_paths('L1C')
        bucket = read_csvfiles_into_lists_of_frames(folders)
        for key, val in bucket.items():
            try:
                df = pd.concat(val, ignore_index=True)
            except ValueError:
                continue
            else:
                df['obsid'] = obsid
                Bucket[key].append(df)
    savedir = pm.path_so_far.parent
    for key, val in Bucket.items():
        try:
            df = pd.concat(val, ignore_index=True)
        except ValueError:
            continue
        else:
            savename = f"{roi_name}_{pm.L1C_folder}_{key}.csv"
            savepath = savedir / savename
            df.to_csv(savepath, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('db_fname',
                        help="Provide the filename of the HDF database "
                             "file here.")
    args = parser.parse_args()

    image_names = io.get_image_names_from_db(args.db_fname)
    logger.info('Found %i image_names', len(image_names))

    c = Client()
    dview = c.direct_view()
    lbview = c.load_balanced_view()

    dview.push({'do_clustering': do_clustering,
                'dbfile': args.db_fname})
    results = lbview.map_async(process_image_name, image_names)
    import time
    import sys
    import os
    dirname = os.path.join(os.environ['HOME'], 'data/planet4/catalog_2_and_3')
    while not results.ready():
        print("{:.1f} %".format(100 * results.progress / len(image_names)))
        sys.stdout.flush()
        time.sleep(10)
    for res in results.result:
        print(res)
    logger.info('Catalog production done. Results in %s.', dirname)
