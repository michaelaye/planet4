
# coding: utf-8

# In[1]:

# setup
get_ipython().magic('matplotlib nbagg')
import seaborn as sns
sns.set_context('notebook')
from planet4 import plotting, io, markings, clustering
from pathlib import Path


# In[64]:

id_ = '1dk'


# In[65]:

cm = clustering.ClusteringManager()


# In[66]:

cm.cluster_image_id(id_)


# In[67]:

plotting.plot_raw_fans(id_)


# In[68]:

plotting.plot_image_id_pipeline(id_)


# In[41]:

plotting.plot_raw_blotches(id_)


# In[38]:

plotting.plot_clustered_fans(id_)


# In[39]:

plotting.plot_clustered_blotches(id_)


# In[40]:

plotting.plot_finals(id_)


# In[ ]:



