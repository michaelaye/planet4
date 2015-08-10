from __future__ import division, print_function

import datetime as dt
import glob
import logging
import os
import platform
import shutil
import sys

import blaze as bz
import matplotlib.image as mplimg
import pandas as pd

from . import helper_functions as hf
from .exceptions import NoFilesFoundError

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

node_name = platform.node().split('.')[0]  # e.g. luna4[.diviner.ucla.edu]
HOME = os.environ["HOME"]
if os.environ['USER'] == 'gapo7695':
    data_root = '/Users/gapo7695/Dropbox/myPy/others/P4_sandbox/databaseP4'
elif node_name.startswith('luna4'):
    data_root = '/raid1/maye/planet4'
else:
    data_root = os.path.join(HOME, 'data/planet4')

done_path = os.path.join(data_root, 'done.h5')

location_target_codes = {'giza': [850],
                         'spider_evolution': [950],
                         'ithaca': [945, 850, 950]}


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


class ResultManager:
    resultpath = os.path.join(data_root, 'catalog_2_and_3')

    def __init__(self, image_name):
        self.image_name = image_name
        self.fanfname = os.path.join(self.resultpath,
                                     image_name + '_reduced_fans.hdf')
        self.blotchfname = os.path.join(self.resultpath,
                                        image_name + '_reduced_blotches.hdf')

    def load_dataframes(self):
        self.fans = pd.read_hdf(self.fanfname, 'df')
        self.blotches = pd.read_hdf(self.blotchfname, 'df')

    def clean_up(self):
        if os.path.exists(self.fanfname):
            os.remove(self.fanfname)
        if os.path.exists(self.blotchfname):
            os.remove(self.blotchfname)


def is_catalog_production_good():
    from pandas.core.index import InvalidIndexError
    dbfile = get_current_database_fname()
    p4data = bz.Data('hdfstore://' + dbfile + '::df')
    image_names = p4data.image_name.distinct()

    not_there = []
    invalid_index = []
    value_error = []
    for image_name in image_names:
        try:
            ResultManager(image_name)
        except InvalidIndexError:
            invalid_index.append(image_name)
        except ValueError:
            value_error.append(image_name)
        except:
            not_there.append(image_name)
    if len(value_error) == 0 and len(not_there) == 0 and\
            len(invalid_index) == 0:
        return True
    else:
        return False


def get_subframe(url):
    """Download image if not there yet and return numpy array.

    Takes a data record (called 'line'), picks out the image_url.
    First checks if the name of that image is already stored in
    the image path. If not, it grabs it from the server.
    Then uses matplotlib.image to read the image into a numpy-array
    and finally returns it.
    """
    targetpath = os.path.join(data_root, 'images', os.path.basename(url))
    if not os.path.exists(os.path.dirname(targetpath)):
        os.makedirs(os.path.dirname(targetpath))
    if not os.path.exists(targetpath):
        logging.info("Did not find image in cache. Downloading ...")
        sys.stdout.flush()
        path = urlretrieve(url)[0]
        logging.debug("Done.")
        shutil.move(path, targetpath)
    else:
        logging.debug("Found image in cache.")
    im = mplimg.imread(targetpath)
    return im


def split_date_from_fname(fname):
    fname = os.path.basename(fname)
    datestr = fname.split('_')[0]
    return [int(i) for i in datestr.split('-')]


def get_dt_from_fname(fname):
    """Return date part of planet 4 database files.

    These files are named yyyy-mm-dd_planet_four_classifications.[csv|h5].
    Hence, this returns the date part for files named like that.
    """
    return dt.datetime(*split_date_from_fname(fname))


def get_latest_file(filenames):
    try:
        retval = filenames[0]
    except IndexError:
        print("No files found.")
        return
    dtnow = get_dt_from_fname(retval)
    for fname in filenames[1:]:
        dt_to_check = get_dt_from_fname(fname)
        if dt_to_check > dtnow:
            dtnow = dt_to_check
            retval = fname
    return retval


def get_current_database_fname(datadir=None):
    if datadir is None:
        datadir = data_root
    h5files = glob.glob(datadir + '/2015*_queryable_cleaned.h5')
    return get_latest_file(h5files)


def get_current_season23_dbase(datadir=None):
    if datadir is None:
        datadir = data_root
    h5files = glob.glob(datadir + '/2015*_seasons2and3.h5')
    return get_latest_file(h5files)
    

def get_test_database():
    return pd.read_hdf(os.path.join(data_root, 'test_db_queryable.h5'),
                       'df')


def get_latest_tutorial_data(datadir=None):
    if datadir is None:
        datadir = data_root

    tut_files = glob.glob(datadir + '/*_tutorials.h5')
    tut_files = [i for i in tut_files if os.path.basename(i)[:4].isdigit()]
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
    return pd.read_hdf(os.path.join(data_root, 'blotch_data.h5'), 'df')


def get_image_from_record(line):
    """Download image if not there yet and return numpy array.

    Takes a data record (called 'line'), picks out the image_url.
    First checks if the name of that image is already stored in
    the image path. If not, it grabs it from the server.
    Then uses matplotlib.image to read the image into a numpy-array
    and finally returns it.
    """
    url = line.image_url
    targetpath = os.path.join(data_root, 'images', os.path.basename(url))
    if not os.path.exists(targetpath):
        print("Did not find image in cache. Downloading ...")
        sys.stdout.flush()
        path = urlretrieve(url)[0]
        print("Done.")
        shutil.move(path, targetpath)
    else:
        print("Found image in cache.")
    im = mplimg.imread(targetpath)
    return im


def get_image_names_from_db(dbfname):
    with pd.HDFStore(dbfname) as store:
        return store.select_column('df', 'image_name').unique()


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
            self.dbname = get_current_database_fname()
        else:
            self.dbname = dbname

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


###
# general database helpers
###

def remove_tutorial(df):
    return df[df.image_name != 'tutorial']
