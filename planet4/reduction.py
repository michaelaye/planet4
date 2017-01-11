#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import progressbar
from ipyparallel import Client

from . import markings
from .io import DBManager, data_root

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# the split trick creates lists when u don't want to break ur fingers with
# typing ',,'','',',,' all the time...
blotch_data_cols = 'x y image_x image_y radius_1 radius_2'.split()
fan_data_cols = 'x y image_x image_y distance angle spread'.split()

analysis_cols = ['classification_id',
                 'created_at',
                 'image_id',
                 'image_name',
                 'image_url',
                 'user_name',
                 'marking',
                 'x_tile',
                 'y_tile',
                 'acquisition_date',
                 'local_mars_time',
                 'x',
                 'y',
                 'image_x',
                 'image_y',
                 'radius_1',
                 'radius_2',
                 'distance',
                 'angle',
                 'spread',
                 'version']

data_columns = ['classification_id', 'image_id',
                'image_name', 'user_name', 'marking',
                'acquisition_date', 'local_mars_time']


def filter_data(df):
    """scan for incomplete data and remove from dataframe.

    This only will work if missing data has been converted to np.NANs.
    """

    # split data into this marking and NOT this marking.
    fans = df[df.marking == 'fan']
    blotches = df[df.marking == 'blotch']
    rest = df[(df.marking != 'fan') & (df.marking != 'blotch')]

    # first drop incomplete data
    fans = fans.dropna(how='any', subset=fan_data_cols)
    blotches = blotches.dropna(how='any', subset=blotch_data_cols)

    # now filter for default data
    eps = 0.00001
    bzero_filter = (blotches.x.abs() < eps) & (blotches.y.abs() < eps)
    fzero_filter = (fans.x.abs() < eps) & (fans.y.abs() < eps)
    rest_zero_filter = (rest.x.abs() < eps) & (rest.y.abs() < eps)
    blotch_defaults = ((blotches.radius_1 - 10) < eps) & ((blotches.radius_2 - 10).abs() < eps)
    fan_defaults = (fans.angle.abs() < eps) & ((fans.distance - 10).abs() < eps)

    fans = fans[~(fzero_filter & fan_defaults)]
    blotches = blotches[~(bzero_filter & blotch_defaults)]
    rest = rest[~rest_zero_filter]

    # merge previously splitted together and return
    df = pd.concat([fans, blotches, rest], ignore_index=True)

    none = df[df.marking == 'none']
    rest = df[df.marking != 'none']

    # filter out markings outside tile frame
    # delta value is how much I allow x and y positions to be outside the
    # planet4 tile
    delta = 25
    q = "{} < x < {} and {} < y < {}".format(-delta, markings.img_x_size + delta,
                                             -delta, markings.img_y_size + delta)

    rest = rest.query(q)
    return pd.concat([rest, none], ignore_index=True)


def convert_times(df):
    logger.info("Starting time conversion now.")
    df.acquisition_date = pd.to_datetime(df.acquisition_date)
    df.created_at = pd.to_datetime(df.created_at,
                                   format='%Y-%m-%d %H:%M:%S %Z')
    logger.info("Time conversions done.")


def splitting_tutorials(rootpath, df):
    logger.info("Splitting off tutorials now.")
    tutorials = df[df.image_name == 'tutorial']
    tutfpath = '{}_tutorials.h5'.format(rootpath)
    tutorials = tutorials.drop(['image_id',
                                'image_url',
                                'image_name',
                                'local_mars_time'], axis=1)
    tutorials.to_hdf(tutfpath, 'df', format='t')

    logger.info("Tutorial split done.\nCreated %s.", tutfpath)
    return df[df.image_name != 'tutorial']


def produce_fast_read(rootpath, df):
    logger.info("Now writing fixed format datafile for "
                "fast read-in of all data.")
    newfpath = '{0}_fast_all_read.h5'.format(rootpath)
    df.to_hdf(newfpath, 'df')
    logger.info("Created %s.", newfpath)


def convert_ellipse_angles(df):
    """Normalize blotch/ellipse angles.

    First we sort for radius_1 to be bigger than radius_2, including
    a correction on the angle, then we do a module 180
    to force the angles to be from 0..180 instead of -180..180.
    This is supported by the symmetry of ellipses.
    """
    logger.info("Converting ellipse angles.")

    blotchindex = (df.marking == 'blotch')
    radindex = (df.radius_1 < df.radius_2)
    both = blotchindex & radindex
    col_orig = ['radius_1', 'radius_2']
    col_reversed = ['radius_2', 'radius_1']
    df.loc[both, col_orig] = df.loc[both, col_reversed].values
    df.loc[both, 'angle'] += 90
    df.loc[blotchindex, 'angle'] = df.loc[blotchindex, 'angle'] % 180
    logger.info("Conversion of ellipse angles done.")


def normalize_fan_angles(df):
    """Convert -180..180 angles to 0..360"""
    logger.info("Normalizing fan angles.")

    rowindex = (df.marking == 'fan')
    df.loc[rowindex, 'angle'] = df.loc[rowindex, 'angle'] % 360
    logger.info("Normalizing of fan angles done.")


def calculate_hirise_pixels(df):
    logger.info("Calculating and assigning hirise pixel coordinates")
    df = df.assign(hirise_x=lambda row: (row.x + 740 * (row.x_tile - 1)).round(),
                   hirise_y=lambda row: (row.y + 548 * (row.y_tile - 1)).round())
    logger.info("Hirise pixels coords added.")
    return df


def get_temp_fname(image_name, root=None):
    if root is None:
        root = data_root
    return str(root / ('temp_' + image_name + '.h5'))


def get_image_names(dbname):
    logger.info('Reading image_names from disk.')
    store = pd.HDFStore(dbname)
    image_names = store.select_column('df', 'image_name').unique()
    logger.info('Got image_names')
    return image_names


def get_cleaned_dbname(dbname):
    dbname = Path(dbname)
    newname = dbname.stem + '_cleaned' + dbname.suffix
    return dbname.with_name(newname)


def merge_temp_files(dbname, image_names=None):
    logger.info('Merging temp files manually.')

    if image_names is None:
        image_names = get_image_names(dbname)

    dbnamenew = get_cleaned_dbname(dbname)
    logger.info('Creating concatenated db file %s', dbnamenew)
    df = []
    for image_name in image_names:
        try:
            df.append(pd.read_hdf(get_temp_fname(image_name, dbname.parent), 'df'))
        except OSError:
            continue
        else:
            os.remove(get_temp_fname(image_name, dbname.parent))
    df = pd.concat(df, ignore_index=True)

    df.to_hdf(str(dbnamenew), 'df',
              format='table',
              data_columns=data_columns)
    logger.info('Duplicates removal complete.')
    return dbnamenew


def remove_duplicates(df):
    logger.info('Removing duplicates.')

    image_names = df.image_name.unique()

    def process_image_name(image_name):
        data = df[df.image_name == image_name]
        data = remove_duplicates_from_image_name_data(data)
        data.to_hdf(get_temp_fname(image_name), 'df')

    # parallel approach, u need to launch an ipcluster/controller for this work!
    lbview = setup_parallel()
    lbview.map_sync(process_image_name, image_names)

    df = []
    for image_name in image_names:
        try:
            df.append(pd.read_hdf(get_temp_fname(image_name), 'df'))
        except OSError:
            continue
        else:
            os.remove(get_temp_fname(image_name))
    df = pd.concat(df, ignore_index=True)
    logger.info('Duplicates removal complete.')
    return df


def display_multi_progress(results, objectlist, sleep=1):
    with progressbar.ProgressBar(min_value=0, max_value=len(list(objectlist)) - 1) as bar:
        while not results.ready():
            bar.update(results.progress)
            time.sleep(sleep)


def setup_parallel(dbname):
    c = Client()
    dview = c.direct_view()
    dview.push({'dbname': str(dbname)})
    # dview.push({'remove_duplicates_from_image_name_data':
    #             remove_duplicates_from_image_name_data,
    #             'get_temp_fname': get_temp_fname,
    #             'dbname': dbname})
    lbview = c.load_balanced_view()
    return lbview


def remove_duplicates_from_image_name_data(data):
    """remove duplicates from this data.

    Parameters
    ==========
    data: pd.DataFrame
        data filtered for one image_name

    Returns
    =======
    For each `user_name` and `image_id` found in `data` return only the data
    for the first found classification_id. There *should* only be one
    classification_id per user_name and image_id, but sometimes the queue
    presented the same image_id more than once to the same users. This removes
    any later in time classification_ids per user_name and image_id.
    """

    c_ids = []

    def process_user_group(g):
        c_ids.append(g[g.created_at == g.created_at.min()].classification_id.min())

    data.groupby(['image_id', 'user_name'], sort=False).apply(process_user_group)
    return data.set_index('classification_id').loc[set(c_ids)].reset_index()


def remove_duplicates_from_file(dbname):
    logger.info('Removing duplicates.')

    image_names = get_image_names(dbname)
    dbname = Path(dbname)

    def process_image_name(image_name):
        from pandas import read_hdf
        # the where string fishes `image_name` from this scope
        data = read_hdf(dbname, 'df', where='image_name=image_name')
        tmp = remove_duplicates_from_image_name_data(data)
        # data.to_hdf(get_temp_fname(image_name, dbname.parent), 'df')
        return tmp

    # parallel approach, u need to launch an ipcluster/controller for this work!
    lbview = setup_parallel(dbname)
    logger.info('Starting parallel processing.')
    results = lbview.map_async(process_image_name, image_names)
    display_multi_progress(results, image_names)
    logger.info('Done clean up. Now concatenating results.')
    all_df = pd.concat(results, ignore_index=True)
    logger.info("Writing cleaned database file.")
    all_df.to_hdf(get_cleaned_dbname(dbname), 'df', format='table', data_columns=data_columns)
    # merge_temp_files(dbname, image_names)
    logger.info("Done.")


def create_season2_and_3_database():
    """Define season columns and write out seaon 2 and 3 database results.

    Has to be executed after the main reduction has finished.
    Installed as main command line script under name create_season2and3.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("db_fname",
                        help="path to database to be used.")
    args = parser.parse_args()
    logger.info('Starting production of season 2 and 3 database.')
    # read data for season2 and 3
    db = DBManager(args.db_fname)
    season23_image_names = db.season2and3_image_names
    where = "image_name in {}".format(list(season23_image_names))
    season23 = pd.read_hdf(db.dbname, 'df', where=where)

    fname_base = os.path.basename(db.dbname)
    root = os.path.dirname(db.dbname)
    fname_no_ext = os.path.splitext(fname_base)[0]
    rootpath = os.path.join(root, fname_no_ext)
    newfname = '{}_seasons2and3.h5'.format(rootpath)
    if os.path.exists(newfname):
        os.remove(newfname)
    season23.to_hdf(newfname, 'df', format='t', data_columns=data_columns)
    logger.info('Finished. Produced %s', newfname)


def read_csv_into_df(fname, chunks=1e6, test_n_rows=None):
    # creating reader object with pandas interface for csv parsing
    # doing this in chunks as its faster. Also, later will do a split
    # into multiple processes to do this.
    reader = pd.read_csv(fname, chunksize=chunks, na_values=['null'],
                         usecols=analysis_cols, nrows=test_n_rows,
                         engine='c')

    # if chunks were None and test_n_rows were given, then I already
    # have the data frame:
    if chunks is None:
        df = reader
    else:
        # read in data chunk by chunk and collect into python list
        data = [chunk for chunk in reader]
        logger.info("Data collected into list.")

        # convert list into Pandas dataframe
        df = pd.concat(data, ignore_index=True)
    logger.info("Conversion to dataframe complete.")
    data = 0
    return df


def main():
    import imp
    try:
        imp.find_module('tables')
    except ImportError:
        print("Please install the PyTables module. It is required.")
        sys.exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_fname',
                        help="Provide the filename of the database "
                             "dump csv-file here.")
    parser.add_argument('--raw_times',
                        help="Do not parse the times into a Python datetime"
                             " object. For the stone-age. ;) Default:"
                             " parse into datetime object.",
                        action='store_true')
    parser.add_argument('--keep_dirt',
                        help="Do not filter for dirty data. Keep everything."
                             " Default: Do the filtering.",
                        action='store_true')
    parser.add_argument('--do_fastread',
                        help='Produce the fast-read database file for'
                             ' complete read into memory.',
                        action='store_true')
    parser.add_argument('--keep_dups',
                        help='Do not remove duplicates from database now '
                             ' (saves time).',
                        action='store_true')
    parser.add_argument('--test_n_rows',
                        help="Set this to do a test parse of n rows",
                        type=int, default=None)
    parser.add_argument('--only_dups',
                        help="Only do the duplicate removal",
                        action='store_true')

    args = parser.parse_args()

    t0 = time.time()
    logger.info("Starting reduction.")

    # creating file paths
    csvfpath = Path(args.csv_fname)
    fname = csvfpath.absolute()
    rootpath = fname.parent / fname.stem
    # path for database:
    newfpath = '{0}_queryable.h5'.format(rootpath)

    if args.only_dups is True:
        remove_duplicates_from_file(newfpath)
        return

    # as chunksize and nrows cannot be used together yet, i switch chunksize
    # to None if I want test_n_rows for a small test database:
    if args.test_n_rows is not None:
        chunks = None
    else:
        chunks = 1e6

    # first read CSV into dataframe
    df = read_csv_into_df(fname, chunks, args.test_n_rows)

    # convert times to datetime object
    if not args.raw_times:
        convert_times(df)

    # split off tutorials
    df = splitting_tutorials(rootpath, df)

    logger.info('Scanning for and dropping empty lines now.')
    df = df.dropna(how='all')
    logger.info("Dropped empty lines.")

    if not args.keep_dirt:
        logger.info("Now filtering for unwanted data.")
        df = filter_data(df)
        logger.info("Done removing incompletes.")

    convert_ellipse_angles(df)
    normalize_fan_angles(df)

    # calculate x_angle and y_angle for clustering on angles
    df = df.assign(x_angle=np.cos(np.deg2rad(df['angle'])),
                   y_angle=np.sin(np.deg2rad(df['angle'])))

    if args.do_fastread:
        produce_fast_read(rootpath, df)

    logger.info("Now writing query-able database file.")
    df.to_hdf(newfpath, 'df',
              format='table',
              data_columns=['image_name'])
    logger.info("Writing to HDF file finished. Created %s. "
                "Reduction complete.", newfpath)

    # free memory
    df = 0

    if not args.keep_dups:
        remove_duplicates_from_file(newfpath)

    dt = time.time() - t0
    logger.info("Time taken: %f minutes.", dt / 60.0)


if __name__ == '__main__':
    main()
