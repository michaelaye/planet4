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
        date = str(self.name)[:10]
        self.date = dt.datetime(*[int(i) for i in date.split('-')])

    def __getattr__(self, name):
        "looking up things in the Path object if not in `self`."
        return getattr(self.p, name)


def get_latest_file(filenames):
    fnames = list(filenames)
    if len(fnames) == 0:
        raise NoFilesFoundError
    retval = P4DBName(fnames[0])
    dtnow = retval.date
    for fname in fnames[1:]:
        dt_to_check = P4DBName(fname).date
        if dt_to_check > dtnow:
            dtnow = dt_to_check
            retval = P4DBName(fname)
    return retval.p


def get_current_database_fname(datadir=None):
    datadir = data_root if datadir is None else Path(datadir)
    h5files = datadir.glob('201*_queryable_cleaned.h5')
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
    if path.suffix in ['.hdf', '.h5']:
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


class PathManager(object):
    """Manage file paths and folders related to the analysis pipeline.

    Parameters
    ----------
    datapath : str or path.Path
        the base path from where to manage all derived paths. No default assumed
        to prevent errors.
    id_ : str
        The data item id that is used to determine sub-paths
    suffix : {'.hdf', '.h5', '.csv'}
        The suffix that controls the reader function to be used.
    """
    def __init__(self, datapath=None, id_=None, suffix='.hdf'):
        self.id_ = id_
        if datapath is None:
            datapath = get_current_database_fname().parent
        self.fnotched_dir = Path(datapath)
        self.suffix = suffix

        # point reader to correct function depending on required suffix
        if suffix in ['.hdf', '.h5']:
            self.reader = pd.read_hdf
        elif suffix == '.csv':
            self.reader = pd.read_csv

    def setup_folders(self):
        "Setup folder paths and create them if required."
        fnotched_dir = self.fnotched_dir
        if fnotched_dir is None:
            fnotched_dir = Path(data_root) / 'cluster_manager_output'
        else:
            fnotched_dir = Path(fnotched_dir)
        fnotched_dir.mkdir(exist_ok=True)
        self.fnotched_dir = fnotched_dir

        # storage path for the clustered data before fnotching
        output_dir_clustered = fnotched_dir / 'just_clustering'
        output_dir_clustered.mkdir(exist_ok=True)
        self.output_dir_clustered = output_dir_clustered

    def create_cut_folder(self, cut):
        # storage path for the final catalog after applying `cut`
        cut_dir = self.fnotched_dir / 'applied_cut_{:.1f}'.format(cut)
        cut_dir.mkdir(exist_ok=True)
        self.cut_dir = cut_dir
        return cut_dir

    def create_path(self, marking, path):
        try:
            retval = path / (self.id_ + '_' + str(marking) + self.suffix)
        except TypeError as e:
            if 'NoneType' in e.args[0]:
                raise TypeError("self.id_ needs to be set first")
            else:
                raise e
        return retval

    def get_df(self, fpath):
        try:
            return self.reader(str(fpath))
        except OSError:
            return None

    @property
    def fanfile(self):
        return self.create_path('fans', self.fnotched_dir)

    @property
    def reduced_fanfile(self):
        return self.create_path('fans', self.output_dir_clustered)

    @property
    def fandf(self):
        return self.get_df(self.fanfile)

    @property
    def blotchfile(self):
        return self.create_path('blotches', self.fnotched_dir)

    @property
    def reduced_blotchfile(self):
        return self.create_path('blotches', self.output_dir_clustered)

    @property
    def blotchdf(self):
        return self.get_df(self.blotchfile)

    @property
    def fnotchfile(self):
        return self.create_path('fnotches', self.fnotched_dir)

    @property
    def fnotchdf(self):
        return self.get_df(self.fnotchfile)


class DBManager(object):

    """Access class for database activities.

    Provides easy access to often used data items.

    Parameters
    ----------
    dbname : str, optional
        Path to database file to be used. Default: use get_current_database_fname() to
        find it.

    Attributes
    ----------
    image_names
    image_ids
    n_image_ids
    n_image_names
    obsids : Alias to image_ids
    season2and3_image_names

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
        """Return list of unique obsids used in database.

        See also
        --------
        get_image_names_from_db
        """
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
        "numpy.array : List of image_names for season 2 and 3."
        from .helper_functions import define_season_column
        image_names = self.image_names
        metadf = pd.DataFrame(pd.Series(image_names).astype('str'), columns=['image_name'])
        define_season_column(metadf)
        return metadf[(metadf.season > 1) & (metadf.season < 4)].image_name.unique()

###
# general database helpers
###


def remove_tutorial(df):
    return df[df.image_name != 'tutorial']
