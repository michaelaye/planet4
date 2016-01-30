
# coding: utf-8

# In[ ]:

cd ~/Dropbox/CTX_to_jpg/pipeline_check/


# In[ ]:

from glob import glob


# In[ ]:

fnames = glob('*.pdf')


# In[ ]:

ids = [i.split('_')[0] for i in fnames]
ids[:5]


# In[ ]:

from planet4 import clustering
from pathlib import Path


# In[ ]:

path = Path("/Users/klay6683/data/planet4/pipelinecheck2")
cm = clustering.ClusteringManager(fnotched_dir=path,
                                 include_angle=True, include_distance=False, 
                                 include_radius=False, eps=10, min_distance=20)


# In[ ]:

from planet4 import plotting


# In[ ]:

from ipyparallel import Client
c = Client()


# In[ ]:

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


# In[ ]:

get_ipython().magic('matplotlib nbagg')


# In[ ]:

plotting.plot_image_id_pipeline('1c5', datapath=path)


# In[ ]:

plotting.plot_finals('1c5', dir=path)


# In[ ]:

plotting.plot_raw_blotches('1ab')


# In[ ]:

x, y = (282.288, 226.657)
x2, y2 = (303.627, 232.805)


# In[ ]:

from numpy.linalg import norm


# In[ ]:

norm(np.array([x,y]-np.array([x2,y2])))


# In[ ]:

from planet4 import io, helper_functions as hf


# In[ ]:

db = io.DBManager()


# In[ ]:

data = db.get_image_id_markings('1fo')


# In[ ]:

data.classification_id.nunique()


# In[ ]:

data.head()


# In[ ]:

data.columns


# In[ ]:

hf.define_season_column(data)


# In[ ]:

db.get_classification_id_data('50ef41ea95e6e42e89000001')


# In[ ]:

db.dbname


# In[ ]:

lbview = c.load_balanced_view()


# In[ ]:

results = lbview.map_async(process_imgid, ids[20:])


# In[ ]:

for res in results:
    print(res)


# In[ ]:

len(ids)


# In[ ]:

a='50ef41ea95e6e42e89000001'


# In[ ]:

b = '50ef419195e6e40eac000001'


# In[ ]:

a < b


# In[ ]:



