
# coding: utf-8

# In[ ]:

# At the beginning of the notebook
import logging
logger = logging.getLogger()
assert len(logger.handlers) == 1
handler = logger.handlers[0]
handler.setLevel(logging.DEBUG)


# In[ ]:

cd ~/Dropbox/CTX_to_jpg/pipeline_check/


# In[ ]:

from glob import glob

fnames = glob('*.pdf')

ids = [i.split('_')[0] for i in fnames]
ids[:5]


# In[ ]:

from planet4 import clustering, io, markings, helper_functions as hf
from pathlib import Path


# In[ ]:

path = Path("/Users/klay6683/Dropbox/data/planet4/ESP_012322_0985")


# In[ ]:

path.mkdir(exist_ok=True)


# In[ ]:

db = io.DBManager()
data = db.get_image_name_markings('ESP_012322_0985')


# In[ ]:

cm = clustering.ClusteringManager(fnotched_dir=path, include_angle=True, include_distance=False,
                                  include_radius=False, eps=10, min_distance=20)


# In[ ]:

cm.cluster_image_name("ESP_012322_0985", data=data)


# In[ ]:

unique_image_ids = data.image_id.unique()


# In[ ]:

len(unique_image_ids)


# In[ ]:

s = pd.Series(unique_image_ids)


# In[ ]:

s.map(lambda x: data[data.image_id==x].classification_id.nunique())


# In[ ]:

container = []
for item in s:
    container.append(data[data.image_id==item].classification_id.nunique())


# In[ ]:

get_ipython().magic('matplotlib nbagg')
plt.figure()
plt.hist(np.array(container), 100);


# In[ ]:

path = Path("/Users/klay6683/data/planet4/pipelinecheck3")
path.mkdir(exist_ok=True)
cm = clustering.ClusteringManager(fnotched_dir=path,
                                 include_angle=True, include_distance=False, 
                                 include_radius=False, eps=10, min_distance=20)


# In[ ]:

from planet4 import plotting


# In[ ]:

from ipyparallel import Client
c = Client()


# In[ ]:

get_ipython().magic('matplotlib None')


# In[ ]:

def process_imgid(id_):
    d = {}
    d['image_id'] = id_
    import matplotlib.pyplot as plt
    from planet4 import plotting, io, clustering
    from pathlib import Path
    path = Path("/Users/klay6683/data/planet4/pipelinecheck3")
    cm = clustering.ClusteringManager(fnotched_dir=path,
                                 include_angle=True, include_distance=False, 
                                 include_radius=False, eps=10, min_distance=20)
    cm.cluster_image_id(id_)
    db = io.DBManager()
    nunique = db.get_image_id_markings(id_).classification_id.nunique()
    d['nunique'] = nunique
    plotting.plot_image_id_pipeline(id_, title_text="# of class_ids: {}".format(nunique),
                                    datapath=path, save=True)

    plt.close('all')
    return d


# In[ ]:

imid='1a0'


# In[ ]:

process_imgid(imid)


# In[ ]:

lbview = c.load_balanced_view()


# In[ ]:

from nbtools import display_multi_progress


# In[ ]:

todo = ids


# In[ ]:

results = lbview.map_async(process_imgid, todo)


# In[ ]:

display_multi_progress(results, todo)


# In[ ]:

df = pd.DataFrame(results.result)


# In[ ]:

df.head()


# In[ ]:

get_ipython().magic('matplotlib nbagg')


# In[ ]:

df.nunique.hist(bins=20)


# In[ ]:

smaller = df[df.nunique < 40]


# In[ ]:

larger = df[(df.nunique > 95)& (df.nunique < 100)]


# In[ ]:

path


# In[ ]:

smaller_dir = path / 'under_40_class'
smaller_dir.mkdir(exist_ok=True)
larger_dir = path / 'between_95_and_100'
larger_dir.mkdir(exist_ok=True)


# In[ ]:

smaller_dir


# In[ ]:

import shutil


# In[ ]:

list(path.glob(smaller.image_id.iloc[0]+'*.pdf'))


# In[ ]:

for id in smaller.image_id:
    src = list(path.glob(id+'*.pdf'))[0]
    dst = smaller_dir / src.name
    shutil.move(str(src), str(dst))


# In[ ]:

for id in larger.image_id:
    src = list(path.glob(id+'*.pdf'))[0]
    dst = larger_dir / src.name
    shutil.move(str(src), str(dst))


# In[ ]:

for id i

