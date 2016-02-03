
# coding: utf-8

# In[ ]:

# setup
import planet4 as p4
import pandas as pd
from pathlib import Path

from planet4 import clustering, markings, io
data = {'x':[210,211,212], 'y':[100,102, 104], 'image_x':[10010,10011,10012], 
        'image_y':[20000,20002,20004],
        'angle':[34, 37,48], 'radius_1':[20,21,19], 'radius_2':[50,51,52]}
three_blotches_data = pd.DataFrame(data)
blotch = markings.Blotch(three_blotches_data.iloc[0])

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[ ]:

# test_calc_fnotch
actual = clustering.calc_fnotch(4, 4)
assert actual == 0.5
actual = clustering.calc_fnotch(4, 0)
assert actual == 1
actual = clustering.calc_fnotch(0, 4)
assert actual == 0


# In[ ]:

# test_dbscan_xy

# using only x and y (or image_x,image_y)
coords = ['image_x','image_y']
X = three_blotches_data[coords].values
dbscanner = clustering.DBScanner(X, min_samples=2)

assert dbscanner.reduced_data == [[0, 1, 2]]
assert dbscanner.n_rejected == 0


# In[ ]:

# test_dbscan_xy_angle

coords = ['image_x','image_y', 'angle']
X = three_blotches_data[coords].values
dbscanner = clustering.DBScanner(X, min_samples=2)

assert dbscanner.reduced_data == [[0, 1]]
assert dbscanner.n_rejected == 1


# # Testing test database

# In[ ]:

# test_test_database

import pkg_resources as pr

with pr.resource_stream('planet4', 'data/test_db.csv') as f:
    data = pd.read_csv(f)
imid1 = 'APF000012w'
imid2 = 'APF000012q'
vc = data.image_id.value_counts()
assert vc[imid1] == 408
assert vc[imid2] == 133

imid1data = data[data.image_id==imid1]
imid2data = data[data.image_id==imid2]

from tempfile import TemporaryDirectory

_tdir = TemporaryDirectory()
tdir = Path(_tdir.name)
cm = clustering.ClusteringManager(fnotched_dir=tdir)

_id = imid1
cm.cluster_image_id(_id, data=data[data.image_id==_id])

for marking in ['blotches', 'fans', 'fnotches']:
    for ftype in ['.csv', '.hdf']:
        expected = tdir / (_id + '_' + marking + ftype)
        assert expected.exists()

        
_id = imid2
cm.cluster_image_id(_id, data=data[data.image_id==_id])
for marking in ['blotches']:
    for ftype in ['.csv', '.hdf']:
        expected = tdir / (_id + '_' + marking + ftype)
        if marking == 'blotches':
            assert expected.exists()
        else: # 12q,i.e. imdid2 only has blotches
            assert not expected.exists()

expected = tdir / 'applied_cut_0.5'
assert expected.exists() and expected.is_dir()
expected = tdir / 'just_clustering'    

_tdir.cleanup()


# In[ ]:



