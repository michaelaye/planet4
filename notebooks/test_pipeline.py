
# coding: utf-8

# In[1]:

from planet4 import clustering, io, markings, helper_functions as hf, plotting
from pathlib import Path


# In[ ]:

cm.cluster_image_name('ESP_022699_0985')


# In[ ]:

db = io.DBManager()


# In[ ]:

data = db.get_image_name_markings('ESP_022699_0985')


# In[ ]:

from ipyparallel import Client
c = Client()


# In[3]:

# At the beginning of the notebook
import logging
logger = logging.getLogger()
assert len(logger.handlers) == 1
logger.addHandler(logging.StreamHandler())
handler = logger.handlers[1]
handler.setLevel(logging.DEBUG)


# In[2]:

def process_imgid(id_, dynamic=False, angle=False, distance=False, radius=False):
    import matplotlib.pyplot as plt
    from planet4 import plotting, clustering
    from pathlib import Path
    path = Path("/Users/klay6683/data/planet4/pipelinecheck5")
    cm = clustering.ClusteringManager(output_dir=path,
                                      do_dynamic_min_samples=dynamic,
                                      include_angle=angle,
                                      include_distance=distance,
                                      include_radius=radius)
    cm.cluster_image_id(id_)
    plotting.plot_image_id_pipeline(id_, datapath=path, save=True)
#     plt.close('all')
    return id_


# In[4]:

get_ipython().magic('matplotlib inline')


# In[5]:

process_imgid('1dr')


# In[6]:

plotting.plot_clustered_fans('1dr')


# In[7]:

plotting.plot_clustered_blotches('1dr')


# In[8]:

plotting.plot_finals('1dr')


# In[9]:

from planet4.plotting import plot_clustered_blotches, plot_clustered_fans, plot_finals


# In[10]:

image_id = '1dr'
imgid = markings.ImageID(image_id)
fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
axes = axes.ravel()
for ax in axes:
    imgid.show_subframe(ax=ax)
plot_clustered_fans(image_id, ax=axes[0])
plot_clustered_blotches(image_id, ax=axes[1])


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[12]:

process_imgid('1dr', dynamic=True, angle=True, distance=True)


# In[ ]:

p1 = (221.79, 508.936)
p2 = (232, 517)


# In[ ]:

from scipy.linalg import norm


# In[ ]:

dp = np.array(p1) - np.array(p2)


# In[ ]:

dp


# In[ ]:

norm(dp)


# In[ ]:

recheck_ids = ['1dn','1k3','1e4','1fe','1aa','225','1pr','19g']
for imid in recheck_ids:
    print(imid)
    cm.cluster_image_id(imid)
    plotting.plot_image_id_pipeline(imid, datapath=path, save=True)


# In[ ]:

db = io.DBManager()
data = db.get_image_id_markings('1fe')


# In[ ]:

data.classification_id.nunique()


# In[ ]:

plotting.plot_finals(imid, _dir=path)


# In[ ]:

plotting.plot_raw_blotches(imid)


# In[ ]:

from planet4.plotting import blotches_all, fans_all


# In[ ]:

import seaborn as sns
sns.set_context('notebook')
blotches_all(imid)


# In[ ]:

fans_all(imid)


# In[ ]:

lbview = c.load_balanced_view()


# In[ ]:

import nbtools.multiprocessing as mptools


# In[ ]:

results = lbview.map_async(process_imgid, ids)


# In[ ]:

mptools.nb_progress_display(results, ids)


# In[ ]:



