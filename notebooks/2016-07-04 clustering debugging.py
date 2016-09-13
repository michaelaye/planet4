
# coding: utf-8

# In[ ]:

get_ipython().magic('matplotlib nbagg')
from planet4 import io, clustering, plotting, reduction


# In[ ]:

def cluster_and_plot(image_id):
    from planet4 import plotting, clustering
    cm = clustering.ClusteringManager(scope='planet4',
                                      do_dynamic_min_samples=False,
                                      include_angle=True,
                                      quiet=False)
    cm.cluster_image_id(image_id)
    plotting.plot_image_id_pipeline(image_id,
                                    save=True, 
                                    datapath=cm.pm.datapath,
                                    savetitle='no_dynamic',
                                    figtitle="min_samples: {}".format(cm.min_samples))
    return cm


# In[ ]:

cm = cluster_and_plot('bvc')


# In[ ]:

db = io.DBManager()


# In[ ]:

data = db.get_image_id_markings('bvc')


# In[ ]:

n_class_old = data.classification_id.nunique()
n_class_old


# In[ ]:

# number of classifications that include fan and blotches
f1 = data.marking == 'fan'
f2 = data.marking == 'blotch'
n_class_fb = data[f1 | f2].classification_id.nunique()
n_class_fb


# In[ ]:

data=data[data.marking=='blotch']


# In[ ]:

plotting.plot_raw_blotches('bvc')


# In[ ]:

data['y_R'] = 1000 - data['y']


# In[ ]:

data.plot(kind='scatter', x='x',y='y_R')


# In[ ]:

fx1 = data.x < 400 
fx2 = data.x > 300
fy1 = data.y_R > 300
fy2 = data.y_R < 400


# In[ ]:

data = data.reset_index()


# In[ ]:

data[fx1 & fx2 & fy1 & fy2].angle


# In[ ]:

cm.dbscanner.reduced_data


# In[ ]:

from planet4 import markings


# In[ ]:

clusterdata = data[markings.Blotch.to_average].iloc[cm.dbscanner.reduced_data[1]]


# In[ ]:

clusterdata


# In[ ]:

meandata = clusterdata.mean()
meandata


# In[ ]:

clusterdata.iloc[[0,2,3]]


# In[ ]:

meandata = clusterdata.iloc[[0,2,3]].mean()
meandata


# In[ ]:

from scipy.stats import circmean
meandata.angle = np.rad2deg(circmean(np.deg2rad(clusterdata.angle)))


# In[ ]:

meandata


# In[ ]:

np.rad2deg(circmean(np.deg2rad([1,178])))


# In[ ]:

angles = np.arange(-179, 180, 5)
angles


# In[ ]:

angles % 180


# In[ ]:

alldata = reduction.read_csv_into_df('/Volumes/USB128GB/planet4/2016-05-29_planet_four_classifications.csv')


# In[ ]:



