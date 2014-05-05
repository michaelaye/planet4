import pandas as pd
import os
import sys
import urllib
import shutil
import matplotlib.image as mplimg

data_root = '/Users/maye/data/planet4'
done_path = os.path.join(data_root, 'done.h5')


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


def get_current_database(with_tutorial=False):
    if with_tutorial:
        df = pd.read_hdf('/Users/maye/data/planet4/'
                         '2014-02-02_planet_four_classifications.h5',
                         'df')
    else:
        df = pd.read_hdf('/Users/maye/data/planet4/'
                         '2014-02-02_tutorial_id_removed.h5',
                         'df')
    define_season_column(df)
    return df


def get_current_marked():
    return pd.read_hdf(os.path.join(data_root, 'marked.h5'), 'df')


def get_current_done():
    return pd.read_hdf(done_path, 'df')


def get_and_save_done(df, limit=30):
    counts = classification_counts_per_image(df)
    ids_done = counts[counts >= limit].index
    df[df.image_id.isin(ids_done)].to_hdf(done_path, 'df')


###
### Season related stuff
###

def unique_image_ids_per_season(df):
    return df.image_id.groupby(df.season, sort=False).agg(size_of_unique)


def define_season_column(df):
    thousands = df.image_name.str[5:7].astype('int')
    df['season'] = 0
    df.loc[:, 'season'][df.image_name.str.startswith('PSP')] = 1
    df.loc[:, 'season'][(thousands > 10) & (thousands < 20)] = 2
    df.loc[:, 'season'][thousands > 19] = 3


###
### general database stuff
###

def clean_and_save_database(df):
    df = df[df.image_id != 'APF0000x3t']
    define_season_column(df)
    df.loc[:, 'marking'][df.marking.isnull()] = 'None'
    df.to_hdf('Users/maye/data/planet4/current_cleaned.h5', 'df')


def get_current_cleaned():
    """docstring for get_current_cleaned"""
    return pd.read_hdf('/Users/maye/data/planet4/current_cleaned.h5', 'df')
