
# coding: utf-8

# In[ ]:

# setup
get_ipython().magic('matplotlib nbagg')
import seaborn as sns
sns.set_context('notebook')
from planet4 import plotting, io, markings, clustering
from pathlib import Path


# In[ ]:

id_ = '6t3'


# In[ ]:

cm = clustering.ClusteringManager(include_angle=False)


# In[ ]:

cm.db.dbname


# In[ ]:

cm.cluster_image_id(id_)


# In[ ]:

plotting.plot_raw_fans(id_)


# In[ ]:

plotting.plot_clustered_fans(id_)


# In[ ]:

plotting.plot_clustered_blotches(id_)


# In[ ]:

plotting.plot_finals(id_)


# In[ ]:



