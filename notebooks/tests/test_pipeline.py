
# coding: utf-8

# In[2]:

cd ~/Dropbox/CTX_to_jpg/pipeline_check/


# In[3]:

from glob import glob


# In[3]:

fnames = glob('*.pdf')


# In[4]:

ids = [i.split('_')[0] for i in fnames]
ids[:5]


# In[4]:

from planet4 import clustering
from pathlib import Path


# In[5]:

path = Path("/Users/klay6683/data/planet4/pipelinecheck2")
cm = clustering.ClusteringManager(fnotched_dir=path,
                                 include_angle=True, include_distance=False, 
                                 include_radius=False, eps=10, min_distance=20)


# In[6]:

from planet4 import plotting


# In[24]:

from ipyparallel import Client
c = Client()


# In[7]:

def process_imgid(id_):
    import matplotlib.pyplot as plt
    from planet4 import plotting, clustering
    from pathlib import Path
    path = Path("/Users/klay6683/data/planet4/pipelinecheck2")
    cm = clustering.ClusteringManager(fnotched_dir=path,
                                 include_angle=True, include_distance=False, 
                                 include_radius=False, eps=10, min_distance=20)
    print(id_)
    cm.cluster_image_id(id_)
    plotting.plot_image_id_pipeline(id_, datapath=path)
    plt.savefig(str(path / (id_+"_angle.pdf")))
    plt.close('all')


# In[8]:

get_ipython().magic('matplotlib nbagg')


# In[9]:

plotting.plot_image_id_pipeline('1c5', datapath=path)


# In[10]:

plotting.plot_finals('1c5', dir=path)


# In[29]:

plotting.plot_raw_blotches('1ab')


# In[11]:

x, y = (282.288, 226.657)
x2, y2 = (303.627, 232.805)


# In[12]:

from numpy.linalg import norm


# In[13]:

norm(np.array([x,y]-np.array([x2,y2])))


# In[17]:

from planet4 import io, helper_functions as hf


# In[12]:

db = io.DBManager()


# In[13]:

data = db.get_image_id_markings('1fo')


# In[14]:

data.classification_id.nunique()


# In[15]:

data.head()


# In[16]:

data.columns


# In[18]:

hf.define_season_column(data)


# In[22]:

db.get_classification_id_data('50ef41ea95e6e42e89000001')


# In[23]:

db.dbname


# In[41]:

lbview = c.load_balanced_view()


# In[42]:

results = lbview.map_async(process_imgid, ids[20:])


# In[43]:

for res in results:
    print(res)


# In[12]:

len(ids)


# In[24]:

a='50ef41ea95e6e42e89000001'


# In[25]:

b = '50ef419195e6e40eac000001'


# In[26]:

a < b


# In[ ]:



