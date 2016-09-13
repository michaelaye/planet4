
# coding: utf-8

# In[1]:

# setup
get_ipython().magic('matplotlib nbagg')
import seaborn as sns
sns.set_context('notebook')
from planet4 import plotting, io, markings, clustering
from pathlib import Path


# In[2]:

id = 'ESP_021520_0925'


# In[3]:

db = io.DBManager('/Users/klay6683/data/planet4/2016-05-29_planet_four_classifications_queryable_cleaned_seasons2and3.h5')


# In[4]:

db.dbname


# In[5]:

data = db.get_image_name_markings(id)


# In[6]:

data.image_id.unique()


# In[ ]:

plotting.plot_raw_fans(


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

