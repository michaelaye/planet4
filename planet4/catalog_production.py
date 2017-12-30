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

from . import io

logger = logging.getLogger(__name__)
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
    ## fnotch across all the HiRISE image
    ## does not work yet correctly! Needs to scale for n_classifications
    if fnotch_via_obsid is True:
        fnotching.fnotch_obsid(obsid, savedir=savedir)
        fnotching.apply_cut_obsid(obsid, savedir=savedir)
    else:
        ## default case: Fnotch for each image_id separately.
        fnotching.fnotch_image_ids(obsid, savedir=savedir)
        fnotching.apply_cut(obsid, savedir=savedir)
    return obsid


def process_obsid_parallel(args):
    obsid, savedir = args
    return cluster_obsid(obsid, savedir)


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
            for key in bucket:
                if key in str(markingfile):
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
