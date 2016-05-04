
# coding: utf-8

# In[1]:

from pathlib import Path
from planet4 import clustering, io, markings, region_data


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


# In[ ]:

obsids = region_data.Inca.season2


# In[4]:

def do_parallel_hirise_scope(obsid):
    from planet4 import clustering, io
    path = io.p4data() / ('overlap_issue_GH45/hirise_level/' + obsid)
    path.mkdir(exist_ok=True, parents=True)
    cm = clustering.ClusteringManager(fnotched_dir=path, scope='hirise')
    cm.cluster_image_name(obsid)
    return cm


# In[5]:

cm = do_parallel_hirise_scope(obsid)


# In[6]:

from pathlib import Path
root = io.dropbox()
fname = root / "overlap_issue_GH45/hirise_level/ESP_011394_0935/applied_cut_0.5/"


# In[ ]:




# In[ ]:




# In[ ]:

from ipyparallel import Client
c = Client()

dview = c.direct_view()

lbview = c.load_balanced_view()


# In[ ]:

from nbtools import display_multi_progress


# In[ ]:

results = lbview.map_async(do_parallel_hirise_scope, obsids)


# In[ ]:

display_multi_progress(results, obsids)


# In[ ]:

db = io.DBManager()

data = db.get_image_name_markings(obsid)

image_ids = data.image_id.unique()


# In[ ]:

def do_in_parallel_p4scope(obsid):
    from pathlib import Path
    from planet4 import clustering, io

    db = io.DBManager()
    data = db.get_image_name_markings(obsid)
    image_ids = data.image_id.unique()
    
    path = io.p4data() / ('overlap_issue_GH45/p4_level_p4_coords/' + obsid)
    path.mkdir(exist_ok=True, parents=True)
    cm = clustering.ClusteringManager(fnotched_dir=path, scope='planet4',
                                 include_angle=True, include_distance=False, 
                                 include_radius=False, eps=10, min_distance=10)
    try:
        for image_id in image_ids:
            cm.cluster_image_id(image_id)
    except:
        return obsid, False
    else:
        return obsid, True


# In[ ]:

results = lbview.map_async(do_in_parallel_p4scope, obsids)


# In[ ]:

display_multi_progress(results, obsids)


# ## Copying results into right folder.
# 

# In[ ]:

import shutil

targetpath = Path('/Users/klay6683/Dropbox/data/planet4/p4_analysis/inca_s23_redone')
for obsid in region_data.Inca.season2 + region_data.Inca.season3:
    path = io.p4data() / 'overlap_issue_GH45/hirise_level/' / obsid / 'applied_cut_0.5'
    for f in path.glob('*.csv'):
        src = path / f
        dst = targetpath / src.name
        shutil.copy2(str(src), str(dst))


# In[ ]:




# In[ ]:




# In[ ]:




# ## Comparing stuff

# In[ ]:

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

    @property
    def n_total(self):
        return self.blotches_lengths + self.fans_lengths + self.fnotches_lengths
    
    def __repr__(self):
        s = "Blotches: {}\n".format(self.blotches_lengths)
        s += "Fan: {}\n".format(self.fans_lengths)
        s += "Fnotches: {}\n".format(self.fnotches_lengths)
        s += "Total: {}".format(self.n_total)
        return s
        
    def __str__(self):
        return self.__repr__()
    
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


# In[ ]:

def compare_per_obsid(obsid):
    print(obsid)
    hicomp = HiRISEComparer(obsid)
    p4comp = P4Comparer(obsid)
    b_r = p4comp.blotches_lengths / hicomp.blotches_lengths
    f_r = p4comp.fans_lengths / hicomp.fans_lengths
    fn_r = p4comp.fnotches_lengths / hicomp.fnotches_lengths
    t_r = p4comp.n_total / hicomp.n_total
    return b_r, f_r, fn_r, t_r


# In[ ]:

df = pd.DataFrame(obsids, columns=['obsid'])


# In[ ]:

def get_ratios(row):
    obsid = row.obsid
    hicomp = HiRISEComparer(obsid)
    p4comp = P4Comparer(obsid)
    b_r = p4comp.blotches_lengths / hicomp.blotches_lengths
    f_r = p4comp.fans_lengths / hicomp.fans_lengths
    fn_r = p4comp.fnotches_lengths / hicomp.fnotches_lengths
    t_r = p4comp.n_total / hicomp.n_total
    return pd.Series(dict(blotch=b_r, fan=f_r, fnotch=fn_r, total=t_r, obsid=obsid))


# In[ ]:

results = df.apply(get_ratios, axis='columns')


# In[ ]:

results


# In[ ]:

results.set_index('obsid', inplace=True)
results.sort_index(inplace=True)


# In[ ]:

results


# In[ ]:

results.plot(style='*-', rot=60)


# In[ ]:

results.index


# In[ ]:

df = df[['obsid']]


# In[ ]:

for marking in ['blotches', 'fans', 'fnotches']:
    for compare,cls in zip(['hi', 'p4'], [HiRISEComparer, P4Comparer]):
        colname = compare + '_' + marking
        df[colname] = df.obsid.map(lambda x: getattr(cls(x), marking+'_lengths'))


# In[ ]:

df['hi_total'] = df.obsid.map(lambda x: HiRISEComparer(x).n_total)
df['p4_total'] = df.obsid.map(lambda x: P4Comparer(x).n_total)


# In[ ]:

df.set_index('obsid', inplace=True)


# In[ ]:

df.sort_index(inplace=True)


# In[ ]:

df


# In[ ]:

df.plot(style='*-', rot=90)


# In[ ]:




# In[ ]:

get_ipython().magic('matplotlib nbagg')


# In[ ]:

P4Comparer(obsid)


# In[ ]:

P4Comparer(obsid, fnotched=False)


# In[ ]:

blotches = P4Comparer(obsid).read_dataframes('blotches')


# In[ ]:

blotches.head()


# In[ ]:

from sklearn.metrics.pairwise import pairwise_distances as pdist


# In[ ]:

arr = np.array([[100,100,5],[101,101,6],[102, 101, 10]])
arr


# In[ ]:

distances = pdist(arr)
distances


# In[ ]:

indices = np.triu_indices_from(distances, k=1)
distances[indices]


# In[ ]:

indices


# In[ ]:

a = [100, 100, 10, 20, 45]
b = [101, 101, 11, 21, 49.5]


# In[ ]:

pdist(np.array([a,b]))


# In[ ]:

res = pdist(blotches[['image_x','image_y', 'radius_1', 'radius_2', 'angle']])


# In[ ]:

indices = np.triu_indices_from(res, k=1)


# In[ ]:

upper = res[indices]


# In[ ]:

upper


# In[ ]:

for i in range(1,6):
    print(i, upper[upper<i].shape)


# In[ ]:




# In[ ]:

import seaborn as sns


# In[ ]:

get_ipython().magic('matplotlib inline')


# In[ ]:

sns.jointplot(x='image_x', y='image_y', kind='hex', data=blotches)


# In[ ]:

all_combined = read_combined_df(path)


# In[ ]:

840*648 - (640*448)


# In[ ]:

_/(840*648)


# In[ ]:

_/__


# In[ ]:

all_combined.info()


# In[ ]:

p = io.p4data() / 'overlap_issue_GH45/p4_level_p4_coords/applied_cut_0.5'
get_total_survivors(p)


# In[ ]:



