
# coding: utf-8

# In[ ]:

# setup
get_ipython().magic('matplotlib nbagg')
import seaborn as sns
sns.set_context('notebook')
from planet4 import plotting, io, markings, clustering
from pathlib import Path


# In[ ]:

id_ = '1dk'


# In[ ]:

cm = clustering.ClusteringManager()


# In[ ]:

cm.cluster_image_id(id_)


# In[ ]:

plotting.plot_raw_fans(id_)


# In[ ]:

plotting.plot_image_id_pipeline(id_)


# In[ ]:

plotting.plot_raw_blotches(id_)


# In[ ]:

plotting.plot_clustered_fans(id_)


# In[ ]:

plotting.plot_clustered_blotches(id_)


# In[ ]:

plotting.plot_finals(id_)

