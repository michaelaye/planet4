
# coding: utf-8

# In[ ]:

# setup
import planet4 as p4
from planet4 import clustering, io, markings
import pandas as pd
from pathlib import Path
import pkg_resources as pr

with pr.resource_stream('planet4', 'data/test_db.csv') as f:
    data = pd.read_csv(f)

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    
from tempfile import TemporaryDirectory

_tdir = TemporaryDirectory()
tdir = Path(_tdir.name)

imid1 = 'APF000012w'
imid2 = 'APF000012q'

imid1data = data[data.image_id==imid1]
imid2data = data[data.image_id==imid2]

fans1 = imid1data[imid1data.marking=='fan']
blotches1 = imid1data[imid1data.marking=='blotch']
fans2 = imid2data[imid2data.marking=='fan']
blotches2 = imid2data[imid2data.marking=='blotch']

# basic clustering manager
cm = clustering.ClusteringManager(dbname='ignore', fnotched_dir=tdir)


# In[ ]:

# test_calc_fnotch
actual = clustering.calc_fnotch(4, 4)
assert actual == 0.5
actual = clustering.calc_fnotch(4, 0)
assert actual == 1
actual = clustering.calc_fnotch(0, 4)
assert actual == 0


# In[ ]:

# test_dbscan_xy_blotch

# using only x and y (or image_x,image_y)
coords = ['image_x','image_y']
X = blotches1[coords].values
dbscanner = clustering.DBScanner(X, min_samples=2)

assert dbscanner.n_clusters == 26
assert dbscanner.n_rejected == 25


# In[ ]:

# test_dbscan_xy_fan

# using only x and y (or image_x,image_y)
coords = ['image_x','image_y']
X = fans1[coords].values
dbscanner = clustering.DBScanner(X, min_samples=2)

assert dbscanner.n_clusters == 7
assert dbscanner.n_rejected == 11


# In[ ]:

# test_dbscan_xy_angle_blotch

coords = ['image_x','image_y', 'angle']
X = blotches1[coords].values
dbscanner = clustering.DBScanner(X, min_samples=2)

assert dbscanner.n_clusters == 35
assert dbscanner.n_rejected == 102


# In[ ]:

# test_dbscan_xy_angle_fan

coords = ['image_x','image_y', 'angle']
X = fans1[coords].values
dbscanner = clustering.DBScanner(X, min_samples=2)

assert dbscanner.n_clusters == 6
assert dbscanner.n_rejected == 15


# In[ ]:

# test_clustering_basics

cm.cluster_image_id(imid1, data=imid1data)
assert cm.n_classifications == 94

cm.cluster_image_id(imid2, data=imid2data)
assert cm.n_classifications == 121

for subdir in ['applied_cut_0.5', 'just_clustering']:
    expected = tdir / subdir
    assert expected.exists() and expected.is_dir()


# In[ ]:

# test_output_file_creation

for marking in ['blotches', 'fans', 'fnotches']:
    for ftype in ['.csv']:
        expected = tdir / (imid1 + '_' + marking + ftype)
        assert expected.exists()

for marking in ['blotches']:
    for ftype in ['.csv']:
        expected = tdir / (imid2 + '_' + marking + ftype)
        if marking == 'blotches':
            assert expected.exists()
        else: # 12q,i.e. imdid2 only has blotches
            assert not expected.exists()


# In[ ]:

dbname='/Volumes/USB128GB/planet4/2016-05-29_planet_four_classifications_queryable_cleaned.h5'


# In[ ]:

from planet4 import clustering


# In[ ]:

# image_name = 'ESP_011394_0935'
image_name = 'ESP_011350_0945'
cm = clustering.ClusteringManager(dbname=dbname, fnotched_dir=tdir, scope='hirise')
cm.cluster_image_name(image_name)


# In[ ]:

cm = clustering.ClusteringManager(dbname='ignore', fnotched_dir=tdir, scope='planet4')
cm.cluster_image_id('apx', data=data)


# In[ ]:

p = tdir / 'applied_cut_0.5'


# In[ ]:

def hunt_bug(fname):
    df = pd.read_csv(fname)
    return (df[df.duplicated()].shape[0])


# In[ ]:

fnames = p.glob('*.csv')
obsids = []
no_of_dupes = []
kind = []
for fname in fnames:
    tokens = fname.name.split('_')
    if fname.name.startswith('ESP'):
        obsids.append('_'.join(tokens[:3]))
        kind.append(tokens[3].split('.')[0])
    else:
        obsids.append(tokens[0])
        kind.append(tokens[1].split('.')[0])
    no_of_dupes.append(hunt_bug(fname))


# In[ ]:

df = pd.DataFrame(dict(obsids=obsids, no_of_dupes=no_of_dupes,
                       kind=kind))
df


# In[ ]:

newblotches = cm.newblotches.apply(lambda x: x.store())


# In[ ]:

newblotches[newblotches.duplicated(keep=False)]


# In[ ]:

cm.pm.fnotchdf.head()


# In[ ]:

cm.newblotches.duplicated().value_counts()


# In[ ]:

cm.pm.fnotchdf.filter(regex='_image_id').head()


# In[ ]:

cm.pm.fnotchdf.iloc[2:4].T


# In[ ]:

fn1 = markings.Fnotch.from_series(cm.pm.fnotchdf.iloc[2], scope='hirise')


# In[ ]:

fn2 = markings.Fnotch.from_series(cm.pm.fnotchdf.iloc[3], scope='hirise')


# In[ ]:

fn1.fan


# In[ ]:

fn2.fan


# In[ ]:

norm(fn1.fan.midpoint - fn2.fan.midpoint)


# In[ ]:

p4id = markings.ImageID('apx', data=data, scope='planet4')


# In[ ]:

get_ipython().magic('matplotlib inline')


# In[ ]:

p4id.plot_blotches()


# In[ ]:

from planet4 import plotting


# In[ ]:

plotting.plot_clustered_blotches('apx', _dir=tdir)


# In[ ]:

pm = io.PathManager(id_='apx', datapath=tdir)


# In[ ]:

pm.reduced_blotchfile


# In[ ]:

tdir


# In[ ]:

list(tdir.glob('just_clustering/*'))


# In[ ]:

newblotches = cm.newblotches.apply(lambda x: x.store())


# In[ ]:

newblotches.head()


# In[ ]:

b1 = markings.Blotch(cm.newblotches.iloc[0].data)
b2 = markings.Blotch(cm.newblotches.iloc[1].data)
b1 == b2


# In[ ]:

df = cm.pm.fnotchdf


# In[ ]:

get_ipython().magic('pinfo df.apply')


# In[ ]:

df.duplicated().value_counts()


# In[ ]:

final_clusters = df.apply(markings.Fnotch.from_series, axis=1).apply(lambda x: x.get_marking(0.5))


# In[ ]:

pd.set_option('display.width', 10000)


# In[ ]:

final_clusters.head()


# In[ ]:

df.filter(regex='fan_').head()


# In[ ]:

df.filter(regex='blotch_').head()


# In[ ]:

from planet4 import markings


# In[ ]:

fnotch = markings.Fnotch.from_series(df.iloc[0], scope='planet4')


# In[ ]:

from numpy.linalg import norm


# In[ ]:

norm(fnotch.blotch.center - fnotch.fan.midpoint)


# In[ ]:

def filter_for_fans(x):
            if isinstance(x, markings.Fan):
                return x

def filter_for_blotches(x):
    if isinstance(x, markings.Blotch):
        return x


# In[ ]:

final_clusters.apply(filter_for_blotches)


# In[ ]:

pd.read_csv(cm.pm.final_fanfile).duplicated().value_counts()


# In[ ]:

# teardown
_tdir.cleanup()


# In[ ]:



