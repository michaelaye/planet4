
# coding: utf-8

# In[ ]:

from planet4 import io


# In[ ]:

p = io.analysis_folder() / 'catalog_2016_04_13' / 'applied_cut_0.5'


# In[ ]:

p = io.analysis_folder() / 'catalog_2016_04_13'


# In[ ]:

p


# In[ ]:

def hunt_bug(fname):
    df = pd.read_csv(fname)
    return (df[df.duplicated()].shape[0])


# In[ ]:

fnames = p.glob('*.csv')
obsids = []
no_of_dupes = []
kind = []
for fname in fnames:
    tokens = fname.name.split('_')
    obsids.append('_'.join(tokens[:3]))
    kind.append(tokens[3].split('.')[0])
    no_of_dupes.append(hunt_bug(fname))


# In[ ]:

df = pd.DataFrame(dict(obsids=obsids, no_of_dupes=no_of_dupes,
                       kind=kind))


# In[ ]:

from hirise import hirise_tools as ht


# In[ ]:

df['orbit'] = df.obsids.map(lambda x: int(ht.OBSERVATION_ID(x).orbit))


# In[ ]:

get_ipython().magic('matplotlib inline')


# In[ ]:

df.orbit.plot()


# In[ ]:

df[df.orbit < 18000].head(15)


# # reproduce

# In[ ]:

obsid = 'ESP_011296_0975'


# In[ ]:

fnames = list(p.glob(obsid+'*.csv'))
fnames


# In[ ]:

blotches = pd.read_csv(fnames[0])
fans = pd.read_csv(fnames[1])
# fnotches = pd.read_csv(fnames[2])


# In[ ]:

for item in [blotches,fans]:
    print(item.duplicated().value_counts())


# In[ ]:



