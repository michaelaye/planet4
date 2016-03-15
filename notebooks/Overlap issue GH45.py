
# coding: utf-8

# In[1]:

from pathlib import Path
from planet4 import clustering, io, markings


# In[2]:

# At the beginning of the notebook
import logging
logger = logging.getLogger()
assert len(logger.handlers) == 1
logger.addHandler(logging.StreamHandler())
handler = logger.handlers[1]
handler.setLevel(logging.DEBUG)


# In[3]:

obsid = 'ESP_011394_0935'
#obsid = 'ESP_012821_0865'


# In[17]:

path = io.p4data() / ('overlap_issue_GH45/hirise_level/' + obsid)
path.mkdir(exist_ok=True, parents=True)
cm = clustering.ClusteringManager(fnotched_dir=path,
                                 include_angle=True, include_distance=False, 
                                 include_radius=False)


# In[18]:

cm.cluster_image_name(obsid)


# In[19]:

cm.pm.fnotchfile


# In[ ]:




# In[20]:

db = io.DBManager()


# In[21]:

data = db.get_image_name_markings(obsid)


# In[22]:

image_ids = data.image_id.unique()


# In[23]:

def do_in_parallel(image_id):
    from pathlib import Path
    from planet4 import clustering, io
    path = io.p4data() / ('overlap_issue_GH45/p4_level_p4_coords/' + obsid)
    path.mkdir(exist_ok=True, parents=True)
    cm = clustering.ClusteringManager(fnotched_dir=path, scope='planet4',
                                 include_angle=True, include_distance=False, 
                                 include_radius=False, eps=10, min_distance=10)
    try:
        cm.cluster_image_id(image_id)
    except:
        return image_id, False
    else:
        return image_id, True


# In[24]:

from nbtools import ListProgressBar, display_multi_progress


# In[25]:

from ipyparallel import Client
c = Client()


# In[26]:

dview = c.direct_view()


# In[27]:

dview.push({'obsid':obsid})


# In[28]:

lbview = c.load_balanced_view()


# In[29]:

results = lbview.map_async(do_in_parallel, image_ids)


# In[30]:

display_multi_progress(results, image_ids)


# In[27]:

class Comparer(object):
    markings = ['fans','blotches','fnotches']
    root = io.p4data() / 'overlap_issue_GH45'
    
    def __init__(self, path, fnotched=True):
        if fnotched is True:
            self.path = self.root / path
        else:
            self.path = self.root / path / 'just_clustering'
        
    @property
    def blotches_iter(self):
        return self.path.glob("*_blotches.csv")
    
    @property
    def fans_iter(self):
        return self.path.glob("*_fans.csv")

    @property
    def fnotches_iter(self):
        return self.path.glob('*_fnotches.csv')
            
    def get_length(self, theiter):
        return sum([len(pd.read_csv(str(f))) for f in theiter])

    @property
    def blotches_lengths(self):
        return self.get_length(self.blotches_iter)
    
    @property
    def fans_lengths(self):
        return self.get_length(self.fans_iter)
    
    @property
    def fnotches_lengths(self):
        return self.get_length(self.fnotches_iter)

    def get_total_survivors(self):
        blotches = self.blotches_lengths
        fans = self.fans_lengths
        fnotches = self.fnotches_lengths
        total = blotches + fans + fnotches
        print("Blotches: {}\nFan: {}\nFnotches: {}".format(blotches, fans, fnotches))
        print("Total: {}".format(total))
        
    def read_dataframes(self, marking, as_df=True):
        res = [pd.read_csv(str(p)) for p in self.path.glob('*_{}.csv'.format(marking))]
        return pd.concat(res) if as_df else res
    
    def read_combined_df(self):
        combine_all = []
        for marking in self.markings:
            to_combine = read_dataframes(marking)
            combine_all.append(to_combine)
        all_combined = pd.concat(combine_all)
        return all_combined


class HiRISEComparer(Comparer):
    root = io.p4data() / 'overlap_issue_GH45/hirise_level/'

    
class P4Comparer(Comparer):
    root = io.p4data() / 'overlap_issue_GH45/p4_level_p4_coords'


# In[5]:

hicomp = HiRISEComparer(obsid)
hicomp.get_total_survivors()


# In[6]:

HiRISEComparer(obsid, fnotched=False).get_total_survivors()


# In[11]:

#def scatter_plot_obsid_data(path):
df = pd.read_hdf(str(next(path.glob("*blotch*.hdf"))))
   


# In[12]:

get_ipython().magic('matplotlib nbagg')


# In[22]:

P4Comparer(obsid).get_total_survivors()


# In[23]:

P4Comparer(obsid, fnotched=False).get_total_survivors()


# In[28]:

blotches = P4Comparer(obsid).read_dataframes('blotches')


# In[29]:

blotches.head()


# In[65]:

import seaborn as sns


# In[66]:

get_ipython().magic('matplotlib inline')


# In[68]:

sns.jointplot(x='image_x', y='image_y', kind='hex', data=blotches)


# In[25]:

all_combined = read_combined_df(path)


# In[30]:

840*648 - (640*448)


# In[31]:

_/(840*648)


# In[29]:

_/__


# In[26]:

all_combined.info()


# In[13]:

p = io.p4data() / 'overlap_issue_GH45/p4_level_p4_coords/applied_cut_0.5'
get_total_survivors(p)


# In[ ]:



