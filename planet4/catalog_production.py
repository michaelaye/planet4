"""This script requires to launch a local ipcontroller. If you execute this
locally, do it with `ipcluster start`.
"""
import argparse
import logging

from ipyparallel import Client
from ipyparallel.util import interactive

from .io import DBManager, get_image_names_from_db

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


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
    db = DBManager()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('db_fname',
                        help="Provide the filename of the HDF database "
                             "file here.")
    args = parser.parse_args()

    image_names = get_image_names_from_db(args.db_fname)
    logging.info('Found %i image_names', len(image_names))

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
    logging.info('Catalog production done. Results in %s.', dirname)
