#!/usr/bin/env python
from __future__ import print_function, division
import pandas as pd
import sys
import os
import argparse

def csv_parse(fname):
    pass


def is_okay(row):
    "check for incomplete markings and mark them as not okay."
    if row.marking in ['interesting', 'none']:
        return True
    if row[row.isnull()].shape[0] !=2:
        return False
    else:
        return True


def main(fname, raw_times=False, keep_dirt=False):
    fname_base = os.path.basename(fname)
    root = os.path.dirname(fname)
    fname_no_ext = os.path.splitex(fname_base)[0]
    reader = pd.read_csv(fname, chunksize=1e6, na_values=['null'])
    data = [chunk for chunk in reader]
    df = pd.concat(data, ignore_index=True)
    if not raw_times:
        df.acquisition_date = pd.to_datetime(df.acquisition_date)
    if not keep_dirt:
        df['okay'] = True  # prefill
        df['okay'] = df.apply(is_okay, axis=1)
        df = df[df.okay]
        df.drop('okay', axis=1)
    df.to_hdf(os.path.join(root, fname_no_ext+'.h5'),
              'df')


if __name__ == '__main__':
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
    args = parser.parse_args()
    print(args)
    main(args.csv_fname, args.raw_times, args.keep_dirt)

