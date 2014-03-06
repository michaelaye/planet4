from __future__ import print_function, division
import pandas as pd
import sys
import os
import argparse

def csv_parse(fname):
    pass

def main(fname, parse_times=True, do_filter=True):
    fname_base = os.path.basename(fname)
    root = os.path.dirname(fname)
    fname_no_ext = os.path.splitex(fname_base)[0]
    reader = pd.read_csv(fname, parse_dates=[1], chunksize=1e6, na_values=['null'])
    data = [chunk for chunk in reader]
    df = pd.concat(data, ignore_index=True)
    df.acquisition_date = pd.to_datetime(df.acquisition_date)


    df.to_hdf(os.path.join(root, fname_no_ext+'.h5'),
              'df')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_fname', help="Provide the filename of the database "
                                           "dump csv-file here.")
    args = parser.parse_args()
    print(args.csv_fname)
    # try:
    #     fname = sys.argv[1]
    # except IndexError:
    #     print("Run like so: {} csv-file".format)
    #     sys.exit()
    # parse_times = do_filter = True
    # try:
    #     second_option = sys.argv[2]
    #     if second_option == 'dont_parse_times':
    #         parse_times = False
    #     elif second_option == 'dont_filter':
    #         do_filter = False
    #     else:
    #         print("Second option should be either 'dont_filter' or 'dont_parse_times'.")
    #         sys.exit()
    # except:
    #     pass
    # try:
    #     third_option = sys.argv[3]
    #     if third_option == 'dont_parse_times'    
    # main()