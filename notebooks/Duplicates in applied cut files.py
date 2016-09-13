
# coding: utf-8

# In[ ]:

from planet4 import io


# In[ ]:

p = Path('/Users/klay6683/Dropbox/data/planet4/duplicates_testing/')


# In[ ]:

p = p / 'applied_cut_0.5'
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
i = 0
for fname in fnames:
    i+=1
    tokens = fname.name.split('_')
    obsids.append('_'.join(tokens[:3]))
    kind.append(tokens[3].split('.')[0])
    no_of_dupes.append(hunt_bug(fname))
print(i, 'no of files.')


# In[ ]:

df = pd.DataFrame(dict(obsids=obsids, no_of_dupes=no_of_dupes,
                       kind=kind))
df.sort_values(by='no_of_dupes', ascending=False)


# In[ ]:

df[df.no_of_dupes>0].shape


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

obsid = 'ESP_021520_0925'


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

g = blotches[blotches.duplicated(keep='first')].groupby('image_id')
g.size().sort_values(ascending=False).head(10)


# In[ ]:

from planet4 import plotting,markings


# In[ ]:

image_id = 'APF0000w0x'
blotches_id = blotches[blotches.image_id==image_id]
fans_id = fans[fans.image_id==image_id]


# In[ ]:

blotches_id.head()


# In[ ]:

dbname = '/Users/klay6683/Dropbox/data/planet4/2016-05-29_planet_four_classifications_queryable_cleaned_seasons2and3.h5'
p4id = markings.ImageID(image_id, dbname=dbname, scope='planet4')


# In[ ]:

get_ipython().magic('matplotlib nbagg')


# In[ ]:

p4id.plot_blotches()


# In[ ]:

fig, ax = plt.subplots()
p4id.plot_blotches(blotches=blotches_id, ax=ax)


# In[ ]:

fig, ax = plt.subplots()
p4id.plot_fans(ax=ax)


# In[ ]:

fig, ax = plt.subplots()
p4id.plot_fans(fans=fans_id, ax=ax)


# In[ ]:

from planet4 import clustering,io


# In[ ]:

fname = '/Users/klay6683/data/planet4/2016-05-29_planet_four_classifications_queryable_cleaned.h5'


# In[ ]:

cm = clustering.ClusteringManager(output_dir='duplicates_testing',
                                  dbname=fname)


# In[ ]:

cm.db.dbname


# In[ ]:

db = io.DBManager(dbname=fname)


# In[ ]:

obsiddata = db.get_image_name_markings(obsid)


# In[ ]:

image_id


# In[ ]:

twoids = obsiddata.image_id.unique()[:2]


# In[ ]:

imageids = list(twoids) + [image_id]


# In[ ]:

imageids


# In[ ]:

data = obsiddata[obsiddata.image_id.isin(imageids)]


# In[ ]:

x_tiles = [1,3]
y_tiles = [36, 38]
data = []
for x_tile in x_tiles:
    for y_tile in y_tiles:
        f1 = obsiddata.x_tile == x_tile
        f2 = obsiddata.y_tile == y_tile
        data.append(obsiddata[f1 & f2])
data.append(obsiddata[obsiddata.image_id==image_id])
data = pd.concat(data)


# In[ ]:

data.image_id.unique()


# In[ ]:

cm.cluster_image_name(obsid, data=data)


# In[ ]:

plotting.plot_clustered_fans(datapath='duplicates_testing',image_id=image_id, scope_id=obsid)


# In[ ]:

cm = clustering.ClusteringManager(scope='hirise', output_dir='testing_by_obsid',
                                  dbname=fname)


# In[ ]:

cm.db.dbname


# In[ ]:

cm.cluster_image_name(obsid)


# In[ ]:

plotting.plot_clustered_fans(datapath='testing_by_obsid', image_id=image_id,
                             scope_id=obsid)


# In[ ]:

from hirise import hirise_tools as ht
from pathlib import Path

pid = ht.PRODUCT_ID("PSP_003092_0985_RED")
ht.download_product(Path(pid.jp2_path), saveroot="your_jp2_folder")


# In[ ]:

pid.jp2_path


# In[ ]:



