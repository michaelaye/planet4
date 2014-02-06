import pandas as pd
import sys
import os

fname = sys.argv[1]
fname_base = os.path.basename(fname)
root = os.path.dirname(fname)
fname_no_ext = os.path.splitex(fname_base)[0]

reader = pd.read_csv(fname, parse_dates=[1], chunksize=1e6, na_values=['null','none'])
data = [chunk for chunk in reader]
df = pd.concat(data, ignore_index=True)
df = df.rename(columns=lambda x: 'classification_id' if x.startswith('2013') else x)
df.acquisition_date = pd.to_datetime(df.acquisition_date)
df.to_hdf(os.path.join(root, fname_no_ext+'.h5',)
          'df')
