
# coding: utf-8

# In[ ]:

from planet4 import stats, io
from planet4 import markings


# In[ ]:

with open('./common_golds.txt') as f:
    gold_ids = f.read()
gold_ids = gold_ids.split('\n')
del gold_ids[-1]  # last one is empty


# In[ ]:

len(gold_ids)


# In[ ]:

gold_members = ['michaelaye', 'mschwamb', 'Portyankina', 'CJ-DPI']
gold_plot_colors = list('cmyg')


# In[ ]:

get_ipython().magic('matplotlib inline')


# In[ ]:

gold_id = markings.ImageID(gold_ids[2], scope='planet4')

fig, axes = plt.subplots(2,1, figsize=(12,12))
# markings.set_upper_left_corner(0, 0)
for goldstar, color in zip(gold_members, gold_plot_colors):
    gold_id.plot_fans(user_name=goldstar, ax=axes[0], user_color=color)
    gold_id.plot_blotches(user_name=goldstar, ax=axes[0], user_color=color)
gold_id.plot_fans(ax=axes[1], without_users=gold_members)
gold_id.plot_blotches(ax=axes[1], without_users=gold_members)
axes[0].set_title("Gold stars")
axes[1].set_title("Citizens")
colors = ['k','r','g','r','b']
markings.gold_legend(axes[0])


# # Statistics

# In[ ]:

blotches = data[data.marking == 'blotch']


# In[ ]:

counts = stats.classification_counts_per_image(blotches)


# In[ ]:

superset_ids = counts[counts > 80].index
print(len(superset_ids))
superset = data[data.image_id.isin(superset_ids)]


# In[ ]:

for i in range(5,16):
    ob.plot_blotches(blotches, superset_ids[i])


# In[ ]:

to_save = blotches[blotches.image_id.isin(superset_ids)]


# In[ ]:

to_save.to_hdf('data/blotch_data.h5','df')


# In[ ]:

blotches[blotches.image_id.isin(superset_ids)].to_hdf


# In[ ]:

example_data = blotches[blotches.image_id == superset_ids[4]]


# In[ ]:

superset_ids[4]


# In[ ]:

example_data.to_hdf('data/APF00001ow.h5','df')


# In[ ]:

get_ipython().system('mkdir data')


# In[ ]:

img_id = superset_ids[4]
print(img_id)
ob.plot_blotches(data[data.image_id == img_id], img_id)


# In[ ]:

cols = blotches['x y radius_1 radius_2'.split()]


# In[ ]:

pd.scatter_matrix(cols, diagonal='kde')


# In[ ]:



