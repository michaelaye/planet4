
# coding: utf-8

# In[23]:

from pathlib import Path
from planet4 import clustering, io, markings


# In[2]:

path = Path("/Users/klay6683/data/planet4/overlap_issue_GH45/hirise_level")
path.mkdir(exist_ok=True, parents=True)
cm = clustering.ClusteringManager(fnotched_dir=path,
                                 include_angle=True, include_distance=False, 
                                 include_radius=False, eps=10, min_distance=20)


# In[3]:

obsid = 'ESP_011544_0985'


# In[4]:

cm.cluster_image_name(obsid)


# In[82]:

path = Path("/Users/klay6683/data/planet4/overlap_issue_GH45/p4_level_hirise_coords")
path.mkdir(exist_ok=True, parents=True)
cm = clustering.ClusteringManager(fnotched_dir=path, scope='planet4',
                                 include_angle=True, include_distance=False, 
                                 include_radius=False, eps=10, min_distance=20)


# In[83]:

db = io.DBManager()


# In[84]:

data = db.get_image_name_markings(obsid)


# In[85]:

image_ids = data.image_id.unique()


# In[86]:

image_ids


# In[ ]:

for image_id in image_ids:
    print(image_id)
    cm.cluster_image_id(image_id)


# In[80]:

d = {'a':[4], 'b':[24545]}


# In[81]:

all(d.values())


# In[76]:

all([i in d for i in ['a','b']])


# In[25]:

get_ipython().magic('matp')
lotlib nbagg


# In[26]:

p4id = markings.ImageID('APF00002r3')


# In[27]:

p4id.plot_all()


# In[ ]:



