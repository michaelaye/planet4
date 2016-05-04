
# coding: utf-8

# In[1]:

import planet4 as p4
from planet4 import io, markings


# In[2]:

p4.__version__


# In[3]:

db = io.DBManager('/Users/klay6683/data/planet4/2016-03-27_planet_four_classifications_queryable.h5')
db.dbname


# In[4]:

data = db.get_all()


# In[5]:

data.info()


# In[15]:

get_ipython().magic('matplotlib nbagg')


# In[7]:

data.x.hist(bins=100, log=True)


# In[8]:

from matplotlib import colors


# In[9]:

delta=25


# In[10]:

q = '{} < x < {} and {} < y < {}'.format(-delta, markings.img_x_size + delta,
                                         -delta, markings.img_y_size + delta)


# In[11]:

subdata = data.query(q)


# In[12]:

from pathlib import Path

folder = Path('./plots/new_data_file')
folder.mkdir(exist_ok=True)


# In[13]:

def do_2d_hist(cols, data, bins=500, pretext=''):
    plt.figure(figsize=(8.4, 6.48))
    plt.hist2d(data[cols[0]].fillna(-99), data[cols[1]].fillna(-99),
               cmap='viridis', bins=bins, norm=colors.LogNorm())
    plt.colorbar()
    plt.savefig(str(folder / (pretext + '{}_{}_2dhist.png'.format(cols[0], cols[1]))), dpi=150)


# In[16]:

do_2d_hist(['x','y'], data)


# In[91]:

do_2d_hist('radius_1 radius_2'.split(), blotches, pretext='blotches_')


# In[43]:

data.columns


# In[86]:

do_2d_hist(['distance','angle'], data)


# In[88]:

do_2d_hist('spread angle'.split(), data)


# In[89]:

fans = data[data.marking=='fan']
blotches = data[data.marking=='blotch']


# In[72]:

do_2d_hist('x y'.split(), fans, pretext='fans_')


# In[73]:

do_2d_hist('x y'.split(), blotches, pretext='blotches_')


# In[74]:

do_2d_hist('radius_1 radius_2'.split(), blotches, pretext='blotches_')


# In[90]:

do_2d_hist('spread angle'.split(), fans, pretext='fans_')


# In[ ]:



