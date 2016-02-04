
# coding: utf-8

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

def process_imgid(id_):
    import matplotlib.pyplot as plt
    from planet4 import plotting, clustering
    from pathlib import Path
    path = Path("/Users/klay6683/data/planet4/pipelinecheck3")
    cm = clustering.ClusteringManager(fnotched_dir=path,
                                 include_angle=True, include_distance=False, 
                                 include_radius=False, eps=10, min_distance=20)
    print(id_)
    cm.cluster_image_id(id_)
    plotting.plot_image_id_pipeline(id_, datapath=path, save=True)
    plt.close('all')
    return id_


# In[ ]:

get_ipython().magic('matplotlib nbagg')


# In[ ]:

imid = '1c5'
cm.cluster_image_id(imid)
plotting.plot_image_id_pipeline(imid, datapath=path, save=True)


# In[ ]:

plotting.plot_finals(imid, _dir=path)


# In[ ]:

plotting.plot_raw_blotches(imid)


# In[ ]:

from planet4.plotting import blotches_all, fans_all


# In[ ]:

import seaborn as sns
sns.set_context('notebook')
blotches_all(imid)


# In[ ]:

fans_all(im)


# In[ ]:

lbview = c.load_balanced_view()


# In[ ]:

import nbtools.multiprocessing as mptools


# In[ ]:

results = lbview.map_async(process_imgid, ids)


# In[ ]:

mptools.nb_progress_display(results, ids)


# In[ ]:



