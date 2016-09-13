
# coding: utf-8

# In[ ]:

with open('./common_golds.txt') as f:
    data = f.readlines()


# In[ ]:

data = [i.strip() for i in data]
data[:5]


# In[ ]:

from planet4 import io


# In[ ]:

def cluster_and_plot(image_id):
    from planet4 import plotting, clustering,io
    savepath = io.p4data() / 'common_gold_data_dynamic'
    cm = clustering.ClusteringManager(scope='planet4',
                                      output_dir=savepath,
                                      do_dynamic_min_samples=True,
                                      quiet=True)
    cm.cluster_image_id(image_id)
    print("Clustering done, now plotting.")
    plotting.plot_image_id_pipeline(image_id, save=True, datapath=savepath,
                                    savetitle='dynamic_min_sample',
                                    figtitle="min_samples: {}".format(cm.min_samples))
    return image_id


# In[ ]:

from ipyparallel import Client
c = Client()


# In[ ]:

lbview = c.load_balanced_view()


# In[ ]:

data[:5]


# In[ ]:

cluster_and_plot(data[2])


# In[ ]:

from nbtools import display_multi_progress


# In[ ]:

results = lbview.map_async(cluster_and_plot, data)


# In[ ]:

display_multi_progress(results, data)


# In[ ]:

for res in results:
    print(res)


# In[ ]:

cluster_and_plot(data[0])


# In[ ]:

from planet4 import plotting


# In[ ]:

get_ipython().magic('matplotlib inline')


# In[ ]:

plotting.plot_image_id_pipeline(data[0], datapath=savepath)


# In[ ]:

db = io.DBManager()
db


# In[ ]:



