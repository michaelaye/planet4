
# coding: utf-8

# In[ ]:

from planet4 import io,plotting, clustering


# In[ ]:

db = io.DBManager()
db


# In[ ]:

image_ids = db.image_ids


# In[ ]:

chosen = np.random.choice(image_ids, 200)


# In[ ]:

from pathlib import Path


# In[ ]:

from ipyparallel import Client
c = Client()

lbview = c.load_balanced_view()


# In[ ]:


savepath = Path("/Volumes/Data/planet4/pipeline_p4_scope_checks/new_dynamic_20")
savepath.mkdir(exist_ok=True)
dbname = '/Users/klay6683/Dropbox/data/planet4/2016-05-29_planet_four_classifications_queryable_cleaned_seasons2and3.h5'


# In[ ]:

from planet4.io import get_image_ids_in_folder


# In[ ]:

(savepath.parent / 'plots').glob('*.pdf')


# In[ ]:

image_ids = get_image_ids_in_folder(savepath.parent / 'plots', '.pdf')


# In[ ]:

def cluster_and_plot(image_id):
    from planet4 import plotting, clustering
    cm = clustering.ClusteringManager(dbname=dbname,
                                      scope='planet4',
                                      output_dir=savepath,
                                      min_samples_factor=0.15,
                                      do_dynamic_min_samples=True,
                                      quiet=False)
    cm.cluster_image_id(image_id)
    plotting.plot_image_id_pipeline(image_id, save=True, datapath=savepath,
                                    dbname=dbname, savetitle='new_dynamic_20',
                                    figtitle="min_samples: {}".format(cm.min_samples))
    return image_id


# In[ ]:

all_ids = list(chosen) + image_ids


# In[ ]:

len(image_ids)


# In[ ]:

cluster_and_plot(image_ids[0])


# In[ ]:

results = lbview.map_async(cluster_and_plot, image_ids)


# In[ ]:

display_multi_progress(results, image_ids)


# In[ ]:




# In[ ]:

for res in res


# In[ ]:

np.array(l1+l2).mean()


# In[ ]:

from nbtools import display_multi_progress


# In[ ]:

chosen[:5]


# In[ ]:

np.argwhere(chosen == 'APF0000cag')


# In[ ]:

chosen


# In[ ]:

display_multi_progress(results, chosen)


# In[ ]:

from planet4 import stats


# In[ ]:

db


# In[ ]:

i


# In[ ]:

stats.get_fb_to_all_ratio(db.get_image_id_markings('0nf'))


# In[ ]:

data = db.get_image_id_markings('0nf')


# In[ ]:

data[data.marking=='fan']

