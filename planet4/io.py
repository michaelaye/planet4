from __future__ import division, print_function

import datetime as dt
import logging
import os
import shutil
import sys

import matplotlib.image as mplimg
import pandas as pd
from pathlib import Path

from . import helper_functions as hf
from .exceptions import NoFilesFoundError

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

HOME = Path(os.environ["HOME"])
if os.environ['USER'] == 'gapo7695':
    data_root = Path('/Users/gapo7695/Dropbox/myPy/others/P4_sandbox/databaseP4')
else:
    data_root = HOME / 'Dropbox' / 'data' / 'planet4'

done_path = data_root / 'done.h5'

location_target_codes = {'giza': [850],
                         'spider_evolution': [950],
                         'ithaca': [945, 850, 950]}


def dropbox():
    home = Path.home()
    return home / 'Dropbox' / 'data' / 'planet4'


def get_image_id_data(image_id, feedback=False):
    return pd.read_hdf(get_current_database_fname(), 'df',
                       where='image_id=' + image_id)


def get_image_name_data(image_name, feedback=False):
    if feedback:
        print("Getting current data for image_name {}".format(image_name))
    return pd.read_hdf(get_current_database_fname(), 'df',
                       where='image_name=' + image_name)


def get_list_of_image_names_data(image_names):
    dfs = []
    for image_name in image_names:
        dfs.append(get_image_name_data(image_name))
    return pd.concat(dfs)


def get_subframe(url):
    """Download image if not there yet and return numpy array.

    Takes a data record (called 'line'), picks out the image_url.
    First checks if the name of that image is already stored in
    the image path. If not, it grabs it from the server.
    Then uses matplotlib.image to read the image into a numpy-array
    and finally returns it.
    """
    targetpath = data_root / 'images' / os.path.basename(url)
    targetpath.parent.mkdir(exist_ok=True)
    if not targetpath.exists():
        logging.info("Did not find image in cache. Downloading ...")
        sys.stdout.flush()
        path = urlretrieve(url)[0]
        logging.debug("Done.")
        shutil.move(path, str(targetpath))
    else:
        logging.debug("Found image in cache.")
    im = mplimg.imread(targetpath)
    return im


class P4DBName(object):

    def __init__(self, fname):
        self.p = Path(fname)
        self.name = self.p.name
        self.parent = self.p.parent
        date = str(self.name)[:10]
        self.date = dt.datetime(*[int(i) for i in date.split('-')])


def get_latest_file(filenames):
    fnames = list(filenames)
    retval = P4DBName(fnames[0])
    dtnow = retval.date
    for fname in fnames[1:]:
        dt_to_check = P4DBName(fname).date
        if dt_to_check > dtnow:
            dtnow = dt_to_check
            retval = fname
    return retval.p


def get_current_database_fname(datadir=None):
    if datadir is None:
        datadir = data_root
    h5files = datadir.glob('2015*_queryable_cleaned.h5')
    return get_latest_file(h5files)


def get_current_season23_dbase(datadir=None):
    if datadir is None:
        datadir = data_root
    h5files = list(datadir.glob('2015*_seasons2and3.h5'))
    return DBManager(get_latest_file(h5files))


def get_test_database():
    return pd.read_hdf(str(data_root / 'test_db_queryable.h5'), 'df')


def get_latest_tutorial_data(datadir=None):
    if datadir is None:
        datadir = data_root

    tut_files = datadir.glob('/*_tutorials.h5')
    tut_files = [i for i in tut_files if i.parent[:4].isdigit()]
    if not tut_files:
        raise NoFilesFoundError
    return pd.read_hdf(get_latest_file(tut_files), 'df')


def common_gold_ids():
    # read the common gold_ids to check
    with open('../data/gold_standard_commons.txt') as f:
        gold_ids = f.read()
    gold_ids = gold_ids.split('\n')
    del gold_ids[-1]  # last one is empty
    return gold_ids


def get_example_blotches():
    return pd.read_hdf(str(data_root / 'blotch_data.h5'), 'df')


def get_image_names_from_db(dbfname):
    """Return arrary of HiRISE image_names from database file.

    Parameters
    ----------
    dbfname : pathlib.Path or str
        Path to database file to be used.

    Returns
    -------
    numpy.ndarray
        Array of unique image names.
    """
    path = Path(dbfname)
    if path.suffix in ['.hdf', 'h5']:
        with pd.HDFStore(str(dbfname)) as store:
            return store.select_column('df', 'image_name').unique()
    elif path.suffix == '.csv':
        return pd.read_csv(dbfname).image_id.unique()


def get_current_marked():
    return pd.read_hdf(get_current_database_fname(), 'df',
                       where='marking!=None')


def get_and_save_done(df, limit=30):
    counts = hf.classification_counts_per_image(df)
    ids_done = counts[counts >= limit].index
    df[df.image_id.isin(ids_done)].to_hdf(done_path, 'df')


class DBManager(object):

    """Wrapper class for database file.

    Provides easy access to often used data items.
    """

    def __init__(self, dbname=None):
        """Initialize DBManager class.

        Parameters
        ----------
        dbname : <str>
            Filename of database file to use. Default: Latest produced full
            database.
        """
        if dbname is None:
            self.dbname = str(get_current_database_fname())
        else:
            self.dbname = str(dbname)

    @property
    def image_names(self):
        "Return list of unique obsids used in database."
        return get_image_names_from_db(self.dbname)

    @property
    def image_ids(self):
        "Return list of unique image_ids in database."
        with pd.HDFStore(self.dbname) as store:
            return store.select_column('df', 'image_id').unique()

    @property
    def n_image_ids(self):
        return len(self.image_ids)

    @property
    def n_image_names(self):
        return len(self.image_names)

    @property
    def obsids(self):
        "Alias to self.image_ids."
        return self.image_ids

    def get_obsid_markings(self, obsid):
        "Return marking data for given HiRISE obsid."
        return pd.read_hdf(self.dbname, 'df', where='image_name=' + obsid)

    def get_image_name_markings(self, image_name):
        "Alias for get_obsid_markings."
        return self.get_obsid_markings(image_name)

    def get_image_id_markings(self, image_id):
        "Return marking data for one Planet4 image_id"
        return pd.read_hdf(self.dbname, 'df', where='image_id=' + image_id)

    def get_classification_id_data(self, class_id):
        "Return data for one classification_id"
        return pd.read_hdf(self.dbname, 'df',
                           where="classification_id=='{}'".format(class_id))

    @property
    def season2and3_image_names(self):
        from .helper_functions import define_season_column
        image_names = self.image_names
        metadf = pd.DataFrame(pd.Series(image_names).astype('str'), columns=['image_name'])
        define_season_column(metadf)
        return metadf[(metadf.season > 1) & (metadf.season < 4)].image_name

###
# general database helpers
###


def remove_tutorial(df):
    return df[df.image_name != 'tutorial']
