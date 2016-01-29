
# coding: utf-8

# In[ ]:

# setup
import planet4 as p4
import pandas as pd
from pathlib import Path
datapath = Path(p4.__path__[0]) / 'data'

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

coords = ['image_x','image_y', 'angle']
X = three_blotches_data[coords].values
dbscanner = clustering.DBScanner(X, min_samples=2)

assert dbscanner.reduced_data == [[0, 1]]
assert dbscanner.n_rejected == 1


# In[ ]:

p4id = markings.example_p4id()


# In[ ]:

for eps in range(10,16):
    cm = clustering.ClusteringManager(scope='planet4', eps=eps)
    cm.cluster_image_id(p4id.imgid)
    print(cm.confusion_data)


# # Testing test database

# In[ ]:

data = pd.read_csv(datapath / 'test_db.csv')


# In[ ]:

data.image_id.value_counts()


# In[ ]:

imid1 = 'APF000012w'
imid2 = 'APF000012q'
imid1data = data[data.image_id==imid1]
imid2data = data[data.image_id==imid2]


# In[ ]:

outputdir = Path('/Users/klay6683/Dropbox/data/planet4/test_clustering')
cm = clustering.ClusteringManager(fnotched_dir=outputdir)


# In[ ]:

cm.cluster_image_id(imid1, data=imid1data)


# In[ ]:

from planet4 import plotting
get_ipython().magic('matplotlib nbagg')
import seaborn as sns
sns.set_context('notebook')


# In[ ]:

plotting.plot_raw_fans('12w')


# In[ ]:

from planet4 import markings, plotting
get_ipython().magic('matplotlib nbagg')


# In[ ]:

id_='6t3'
cm = clustering.ClusteringManager(include_angle=False)
cm.cluster_image_id(id_)


# In[ ]:

plotting.plot_image_id_pipeline(id_, include_angle=True, include_distance=True,
                                include_radius=False, eps=10)


# In[ ]:

plotting.plot_image_id_pipeline(id_, include_angle=False, include_distance=True,
                                include_radius=True, eps=10, min_distance=15)


# In[ ]:

plotting.plot_image_id_pipeline(id_, include_angle=False, include_distance=True,
                                include_radius=False, eps=10, min_distance=15)


# In[ ]:

plotting.plot_image_id_pipeline('139', include_angle=True, include_distance=True,
                                include_radius=True, eps=20, min_distance=10)


# In[ ]:

plotting.plot_image_id_pipeline('13t', include_angle=True, include_distance=True,
                                include_radius=True, eps=20, min_distance=10)


# In[ ]:

plotting.plot_finals('13t')


# In[ ]:

plotting.plot_image_id_pipeline('15k', include_angle=True, include_distance=True,
                                include_radius=True, eps=20, min_distance=10)


# In[ ]:

plotting.plot_finals('15k')


# In[ ]:

db = io.DBManager()
data = db.get_image_id_markings('17a')


# In[ ]:

plotting.plot_image_id_pipeline('17a', include_angle=True, include_distance=True,
                                include_radius=False, eps=20, min_distance=20)


# In[ ]:

plotting.plot_raw_fans('17a')


# In[ ]:

plotting.plot_finals('17a')


# In[ ]:

imgid = markings.ImageID(id_)


# In[ ]:

fans = imgid.get_fans()


# In[ ]:

dbscanner = clustering.DBScanner(fans[['x','y']].values)


# In[ ]:

dbscanner.reduced_data


# In[ ]:

fan = fans.iloc[dbscanner.reduced_data[6]]

imgid.plot_fans(fans=markings.FanContainer.from_df(fan).content)


# In[ ]:

fan[['x','y']]


# In[ ]:

markings.Fan.to_average


# In[ ]:

clusterdata = fans[markings.Fan.to_average].iloc[dbscanner.reduced_data[6]]
clusterdata


# In[ ]:

from scipy.stats import circmean


# In[ ]:

np.rad2deg(circmean(np.deg2rad(angles)))


# In[ ]:

meandata = clusterdata.mean()
meandata


# In[ ]:

meandata.angle = np.rad2deg(circmean(np.deg2rad(clusterdata.angle)))


# In[ ]:

meandata


# In[ ]:

type(meandata)


# In[ ]:

imgid.plot_fans(fans=[markings.Fan(meandata)])


# In[ ]:

plotting.plot_clustered_fans('6t3')


# In[ ]:

angle1 = 181
angle2 = 538
fantest = markings.Fan(pd.Series(dict(x=200,y=200, angle=angle1, spread=30, distance=100)),
                       linewidth=2)
fantest2 = markings.Fan(pd.Series(dict(x=200,y=200, angle=angle2, spread=30, distance=100)),
                       linewidth=2)


imgid.plot_fans(fans=[fantest, fantest2])


# In[ ]:

angle2 = 182
fantest2 = markings.Fan(pd.Series(dict(x=meandata.x,y=meandata.y, angle=angle2, 
                                       spread=meandata.spread, distance=meandata.spread)),
                       linewidth=2)
imgid.plot_fans(fans=[fantest2])


# In[ ]:

meandata.angle


# In[ ]:

imgid.plot_fans(fans=[markings.Fan(meandata)])


# In[ ]:



