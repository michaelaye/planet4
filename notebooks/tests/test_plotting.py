
# coding: utf-8

# In[1]:

# setup
get_ipython().magic('matplotlib nbagg')
import seaborn as sns
sns.set_context('notebook')
from planet4 import plotting, io, markings, clustering
from pathlib import Path


# In[2]:

id_ = '6t3'


# In[3]:

cm = clustering.ClusteringManager(include_angle=False)


# In[7]:

cm.db.dbname


# In[4]:

cm.cluster_image_id(id_)


# In[5]:

plotting.plot_raw_fans(id_)


# In[6]:

plotting.plot_clustered_fans(id_)


# In[8]:

plotting.plot_clustered_blotches(id_)


# In[9]:

plotting.plot_finals(id_)


# In[ ]:



