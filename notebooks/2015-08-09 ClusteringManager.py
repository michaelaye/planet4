
# coding: utf-8

# In[ ]:

get_ipython().magic('matplotlib nbagg')
from planet4.clustering import ClusteringManager
from planet4 import io, markings, plotting
from IPython.display import display
from ipywidgets import FloatText
from pathlib import Path


# In[ ]:

db = io.DBManager("/Volumes/USB128GB/planet4/2015-11-22_planet_four_classifications_queryable_cleaned.h5")
dbname = db.dbname


# In[ ]:

obsids = get_ipython().getoutput('cat /Users/klay6683/Dropbox/data/planet4/season2_3_image_names.txt')


# In[ ]:

img_name = obsids[0]


# In[ ]:

def process_image_name(img_name):
    from pathlib import Path
    from planet4.clustering import ClusteringManager
    output_dir = Path('/Users/klay6683/Dropbox/data/planet4/season23_v3')
    output_dir.mkdir(exist_ok=True)
    cm = ClusteringManager(dbname, scope='planet4',
                           output_dir=output_dir,
                           output_format='both')
    cm.cluster_image_name(img_name)
    return img_name


# In[ ]:

process_image_name(img_name)


# In[ ]:

from planet4.clustering import ClusteringManager
fnotched_dir = Path('/Users/klay6683/Dropbox/data/planet4/debug')
fnotched_dir.mkdir(exist_ok=True)


# In[ ]:

def check_processing_image_id(img_id):
    cm = ClusteringManager(dbname, scope='planet4',
                       fnotched_dir=fnotched_dir,
                       output_format='both')
    cm.cluster_image_id(img_id)
    p4id = markings.ImageID(img_id, database_fname=dbname)
    fig, axes = plt.subplots(nrows=2, sharex=True)
    resman = plotting.ResultManager(img_id, fnotched_dir)
    p4id.plot_blotches(ax=axes[0],
                       blotches=resman.blotches(unfnotched=True))
    p4id.plot_fans(ax=axes[1],
                   fans=resman.fans(unfnotched=True))
    cm.apply_fnotch_cut(img_id)
    return cm


# In[ ]:

cm = check_processing_image_id('APF000036k')


# In[ ]:




# In[ ]:

resman = plotting.ResultManager("APF000036k", cm.fnotched_dir)


# In[ ]:

resman.fnotchfile


# In[ ]:

cd ~/Dropbox/data/planet4/debug/


# In[ ]:

ll


# In[ ]:

df = pd.read_hdf('./APF000036k_fnotches.hdf')


# In[ ]:

fc = markings.FnotchContainer.from_df(df)


# In[ ]:

fc.content[0]


# In[ ]:

from ipyparallel import Client
c = Client()


# In[ ]:

lbview = c.load_balanced_view()


# In[ ]:

dview = c.direct_view()


# In[ ]:

dview.push({'dbname': str(db.dbname)})


# In[ ]:

image_ids = []
for image_name in obsids:
    image_ids.append(db.get_image_name_markings(image_name)['image_id'].unique())


# In[ ]:

image_ids = pd.concat([pd.Series(i).astype(str) for i in image_ids])


# In[ ]:

len(image_ids)


# In[ ]:

cm = ClusteringManager(dbname, scope='planet4',
                       output_dir=output_dir,
                       output_format='both', cut=0.5)


# In[ ]:

todo = image_ids[:]
results = lbview.map_async(cm.cluster_image_id, todo)


# In[ ]:

from iuvs.multitools import nb_progress_display


# In[ ]:

nb_progress_display(results, todo)


# In[ ]:

for res in results:
    res


# # Applying cut to fnotches

# In[ ]:

from pathlib import Path
from planet4 import io


# In[ ]:

output_dir = io.p4data() / 'season23_by_id_fnotched_0.5/'
input_dir = io.p4data() / 'season23_by_id'


# In[ ]:

fnotch_fnames = list(input_dir.glob('*_fnotches.hdf'))


# In[ ]:

len(fnotch_fnames)


# In[ ]:

def filter_for_fans(x):
    if isinstance(x, planet4.markings.Fan):
        return x

def filter_for_blotches(x):
    if isinstance(x, planet4.markings.Blotch):
        return x


# In[ ]:

from planet4 import markings, plotting

def get_newfans_newblotches(fname, input_dir):
    resman = plotting.ResultManager(fname, input_dir)
    df = resman.fnotchdf
    final_clusters = df.apply(markings.Fnotch.from_dataframe, axis=1).        apply(lambda x: x.get_marking(0.5))
    newfans = final_clusters[final_clusters.apply(filter_for_fans).notnull()]
    newblotches = final_clusters[final_clusters.apply(filter_for_blotches).notnull()]
    return newfans, newblotches
    
def process_fnotch_fname(fname):
    outpath = output_dir
    outpath.mkdir(exist_ok=True)
    newfans, newblotches = get_newfans_newblotches(fname, input_dir)
    
    if len(newfans) > 0:
        newfans = newfans.apply(lambda x: x.store())
        try:
            completefans = pd.DataFrame(resman.fandf).append(newfans, ignore_index=True)
        except OSError:
            completefans = newfans
    else:
        completefans = resman.fandf
    if len(newblotches) > 0:
        newblotches = newblotches.apply(lambda x: x.store())
        try:
            completeblotches = pd.DataFrame(resman.blotchdf).append(newblotches,
                                                      ignore_index=True)
        except OSError:
            completeblotches = newblotches
    else:
        completeblotches = resman.blotchdf
    completefans.to_hdf(str(outpath / resman.fanfile.name), 'df')
    completeblotches.to_hdf(str(outpath / resman.blotchfile.name), 'df')


# In[ ]:

image_ids = [i.stem.split('_')[0] for i in fnotch_fnames]


# In[ ]:

image_ids[0]


# In[ ]:

cm.apply_fnotch_cut(image_ids[0])


# In[ ]:

results = lbview.map_async(cm.apply_fnotch_cut, image_ids)


# In[ ]:

nb_progress_display(results, image_ids)


# In[ ]:

cm.


# In[ ]:

for res in results:
    res


# In[ ]:

resman = plotting.ResultManager('APF000036k', input_dir)


# In[ ]:

type(resman.fandf)


# In[ ]:

resman.blotchdf


# In[ ]:



