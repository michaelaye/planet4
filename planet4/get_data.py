import pandas as pd
import os
import sys
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import shutil
import glob
import matplotlib.image as mplimg
import datetime as dt
import platform
import logging
from . import helper_functions as hf
import blaze as bz

node_name = platform.node().split('.')[0]  # e.g. luna4[.diviner.ucla.edu]
HOME = os.environ["HOME"]

if os.environ['USER'] == 'gapo7695':
    data_root = '/Users/gapo7695/Dropbox/myPy/others/P4_sandbox/databaseP4'
elif node_name.startswith('luna4'):
    data_root = '/raid1/maye/planet4'
else:
    data_root = os.path.join(HOME, 'data/planet4')
done_path = os.path.join(data_root, 'done.h5')


class ResultManager:
    resultpath = '/Users/klay6683/data/planet4/reduced/'

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
    p4data = bz.Data('hdfstore://'+dbfile+'::df')
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
    retval = filenames[0]
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

    h5files = glob.glob(datadir + '/*_queryable.h5')
    return get_latest_file(h5files)


def get_latest_tutorial_data(datadir=None):
    if datadir is None:
        datadir = data_root

    tut_files = glob.glob(datadir + '/*_tutorials.h5')
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
        path = urllib.urlretrieve(url)[0]
        print("Done.")
        shutil.move(path, targetpath)
    else:
        print("Found image in cache.")
    im = mplimg.imread(targetpath)
    return im


def get_current_marked():
    return pd.read_hdf(get_current_database_fname(), 'df',
                       where='marking!=None')


def get_current_done():
    return pd.read_hdf(done_path, 'df')


def get_and_save_done(df, limit=30):
    counts = hf.classification_counts_per_image(df)
    ids_done = counts[counts >= limit].index
    df[df.image_id.isin(ids_done)].to_hdf(done_path, 'df')


###
### general database stuff
###

def clean_and_save_database(df):
    df = df[df.image_id != 'APF0000x3t']
    hf.define_season_column(df)
    df.loc[:, 'marking'][df.marking.isnull()] = 'None'
    df.to_hdf('Users/maye/data/planet4/current_cleaned.h5', 'df')


def get_current_cleaned():
    """docstring for get_current_cleaned"""
    return pd.read_hdf('/Users/maye/data/planet4/current_cleaned.h5', 'df')
