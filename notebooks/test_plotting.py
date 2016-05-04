
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


# In[94]:

df = pd.DataFrame(5*np.arange(5, dtype='float'))
df


# In[95]:

df.loc[2:4]=df.loc[1:3]


# In[96]:

df


# In[91]:

arr = 5*np.arange(5, dtype='float')


# In[92]:

arr


# In[93]:

arr[2:4]=arr[1:3]
arr


# In[101]:

df = pd.DataFrame(5*np.arange(5, dtype='float'))
df


# In[102]:

df.shift(1)


# In[98]:

df.iloc[1:3]


# In[99]:

df.iloc[2:4]=df.iloc[1:3]


# In[100]:

df


# In[ ]:

df

