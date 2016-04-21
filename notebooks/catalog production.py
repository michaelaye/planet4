
# coding: utf-8

# In[ ]:

from __future__ import print_function, division
import planet4 as p4
import pandas as pd
from planet4 import markings
from planet4.get_data import get_current_database_fname
from planet4 import clustering
import os
from os.path import join as pjoin
HOME = os.environ['HOME']


# In[ ]:

dbfile = get_current_database_fname()
store = pd.HDFStore(dbfile)
store


# In[ ]:

image_names = store.select_column('df', 'image_name').unique()


# In[ ]:

from IPython.parallel import Client
client = Client()


# In[ ]:

dview = client.direct_view()
lview = client.load_balanced_view()


# In[ ]:

get_ipython().run_cell_magic('px', '', "import pandas as pd\nfrom planet4 import clustering, markings\nfrom os.path import join as pjoin\nimport os\nHOME = os.environ['HOME']")


# In[ ]:

def do_clustering(p4img, fans):
    if fans:
        reduced = clustering.perform_dbscan(p4img.get_fans(), fans=fans)
    else:
        reduced = clustering.perform_dbscan(p4img.get_blotches(), fans=fans)
    if reduced is None:
        return None
    series = [cluster.data for cluster in reduced]
    n_members = [cluster.n_members for cluster in reduced]
    df = pd.DataFrame(series)
    df['image_id'] = p4img.imgid
    df['n_members'] = n_members
    return df
    
def process_image_name(image_name):
    dirname = pjoin(HOME, 'data/planet4/reduced')
    blotchfname = pjoin(dirname, image_name+'_reduced_blotches.hdf')
    fanfname = pjoin(dirname, image_name+'_reduced_fans.hdf')
    if os.path.exists(blotchfname) and            os.path.exists(fanfname):
        return image_name+' already done.'
    data = pd.read_hdf(dbfile, 'df', where="image_name="+image_name)
    img_ids = data.image_id.unique()
    blotches = []
    fans = []
    for img_id in img_ids:
        p4img = markings.ImageID(img_id)
        blotches.append(do_clustering(p4img, fans=False))
        fans.append(do_clustering(p4img, fans=True))
    blotches = pd.concat(blotches, ignore_index=True)
    blotches.to_hdf(blotchfname, 'df')
    fans = pd.concat(fans, ignore_index=True)
    fans.to_hdf(fanfname, 'df')
    return image_name


# In[ ]:

dview.push({'do_clustering':do_clustering,
            'dbfile':dbfile})


# In[ ]:

result = lview.map_async(process_image_name, image_names)


# In[ ]:

for res in result:
    print(res)


# In[ ]:

import time
import sys
while not result.ready():
    print("{:.1f} %".format(100*result.progress/len(image_names)))
    sys.stdout.flush()
    time.sleep(30)


# In[ ]:

reducedfiles = get_ipython().getoutput('ls ~/data/planet4/reduced')
nooffiles = len(reducedfiles)
print("Produced", nooffiles, "files.")


# In[ ]:

from planet4.get_data import is_catalog_production_good


# In[ ]:

is_catalog_production_good()


# In[ ]:



