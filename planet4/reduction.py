#!/usr/bin/env python
from __future__ import print_function, division
import pandas as pd
import os
import argparse
import logging
import sys

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# the split trick creates lists when u don't want to break ur fingers with
# typing ',,'','',',,' all the time...
blotch_data_cols = 'x y image_x image_y radius_1 radius_2'.split()
fan_data_cols = 'x y image_x image_y distance angle spread'.split()


def scan_for_incomplete(df, marking):
    """scan for incomplete data and remove from dataframe."""
    # split data into this marking and NOT this marking.
    marked = df[df.marking == marking]
    rest = df[df.marking != marking]

    if marking == 'fan':
        data_cols = fan_data_cols
    elif marking == 'blotch':
        data_cols = blotch_data_cols
    else:
        print("Not supported marking")
        return

    # create index file. basic idea is: from the above defined data cols for
    # a marking, none of them is allowed to be unfilled. (== isnull )
    ind = marked[data_cols].isnull().any(axis=1)
    # select for negated index, as we do NOT want incomplete data
    marked = marked[~ind]

    # merge previously splitted together and return
    return pd.concat([marked, rest])


def main(fname, raw_times=False, keep_dirt=False, do_fastread=False):
    logging.info("Starting reduction.")

    # creating file paths
    fname_base = os.path.basename(fname)
    root = os.path.dirname(fname)
    fname_no_ext = os.path.splitext(fname_base)[0]
    rootpath = os.path.join(root, fname_no_ext)

    # creating reader object with pandas interface for csv parsing
    # doing this in chunks as its faster. Also, later will do a split
    # into multiple processes to do this.
    reader = pd.read_csv(fname, chunksize=1e6, na_values=['null'])

    # read in data chunk by chunk and collect into python list
    data = [chunk for chunk in reader]
    logging.info("Data collected into list.")

    #convert list into Pandas dataframe
    df = pd.concat(data, ignore_index=True)
    logging.info("Conversion to dataframe complete.")

    # convert times to datetime object
    if not raw_times:
        logging.info("Starting time conversion now.")
        df.acquisition_date = pd.to_datetime(df.acquisition_date)
        df.created_at = pd.to_datetime(df.created_at,
                                       format='%Y-%m-%d %H:%M:%S %Z')
        logging.info("Time conversions done.")
    logging.info("Splitting off tutorials now.")
    tutorials = df[df.image_name == 'tutorial']
    tutfpath = '{0}_tutorials.h5'.format(rootpath)
    tutorials.to_hdf(tutfpath, 'df')
    df = df[df.image_name != 'tutorial']

    logging.info("Tutorial split done. Wrote "
                 "{0}. Scanning for and dropping empty lines now."
                 .format(tutfpath))
    df = df.dropna(how='all')
    logging.info("Dropped empty lines.")

    if not keep_dirt:
        logging.info("Now scanning for incomplete marking data.")
        for marking in ['fan', 'blotch']:
            df = scan_for_incomplete(df, marking)
    logging.info("Done removing incompletes.")

    if do_fastread:
        logging.info("Now writing fixed format datafile for "
                     "fast read-in of all data.")
        newfpath = '{0}_fast_all_read.h5'.format(rootpath)
        df.to_hdf(newfpath, 'df')
        logging.info("Created {}.".format(newfpath))

    logging.info("Now writing query-able database file.")
    newfpath = '{0}_queryable.h5'.format(rootpath)
    df.to_hdf(newfpath, 'df',
              format='table',
              data_columns=['classification_id', 'image_id',
                            'image_name', 'user_name', 'marking',
                            'acquisition_date', 'local_mars_time'])

    logging.info("Writing to HDF file finished. Reduction complete.")


if __name__ == '__main__':
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
    args = parser.parse_args()
    main(args.csv_fname, args.raw_times, args.keep_dirt, args.do_fastread)
