
# coding: utf-8

# In[ ]:

from planet4 import io


# In[ ]:

db = io.DBManager('/Users/klay6683/Dropbox/data/planet4/2016-05-29_planet_four_classifications_queryable_cleaned_seasons2and3.h5')


# In[ ]:

db.dbname


# In[ ]:

image_ids = db.image_ids


# In[ ]:

len(image_ids)


# In[ ]:

chosen = np.random.choice(image_ids, 10000)


# In[ ]:

def get_fan_and_blotch_nunique_cids(data):
    f1 = data.marking == 'fan'
    f2 = data.marking == 'blotch'
    return data[f1 | f2].classification_id.nunique()


# In[ ]:

from ipyparallel import Client
c = Client()


# In[ ]:

lbview = c.load_balanced_view()


# In[ ]:

dview = c[:]


# In[ ]:

dview.push({'db': db,
            'get_fan_and_blotch_nunique_cids':get_fan_and_blotch_nunique_cids})


# In[ ]:

def get_ratio(image_id):
    data = db.get_image_id_markings(image_id)
    n_classifications = data.classification_id.nunique()
    n_class_fb = get_fan_and_blotch_nunique_cids(data)
    ratio = (n_class_fb / n_classifications)
    d = {'image_id':image_id, 'ratio':ratio}
    return d


# In[ ]:

results = lbview.map_async(get_ratio, chosen)


# In[ ]:

from nbtools import display_multi_progress


# In[ ]:

display_multi_progress(results, chosen)


# In[ ]:

df = pd.DataFrame(results.result())


# In[ ]:

get_ipython().magic('matplotlib inline')


# In[ ]:

df.ratio.hist(bins=30)


# In[ ]:

import seaborn as sns


# In[ ]:

sns.distplot(df.ratio)


# In[ ]:

t0 = pd.Timestamp('2016-07-01 12:00:00')


# In[ ]:

delta = pd.Timedelta('114h')


# In[ ]:

t0 + delta


# In[ ]:



