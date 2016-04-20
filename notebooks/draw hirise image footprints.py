
# coding: utf-8

# In[40]:

from hirise.indexfiles import PolyPlotter
from hirise import hirise_tools as ht


# In[33]:

from planet4 import region_data, io


# In[4]:

get_ipython().magic('matplotlib inline')


# In[16]:

plotter = PolyPlotter()


# In[44]:

missing = []
regions = ['Inca','Ithaca','Giza','Manhattan2']
seasons = ['season1', 'season2', 'season3', 'season4']
colors = ['blue', 'green','red', 'black']
for region in regions:
    reg = getattr(region_data, region)
    fig, ax = plt.subplots()
    ax.set_title('{}'.format(region))
    handels = []
    labels = []
    for season,c in zip(seasons, colors):
        try:
            seas = getattr(reg, season)
        except AttributeError as e:
            print(e)
            continue
        for img in seas:
            try:
                plotter.plot_prodid(img+'_COLOR', ax=ax, color=c, linewidth=1, alpha=0.5)
            except KeyError as e:
                print(e)
                missing.append(img+'+_COLOR')
    patches = [plt.Line2D((0,1),(0,0), color=c) for c in colors]
    ax.legend(patches, seasons)
    savename = ht.hirise_dropbox() / "{}_polygons.pdf".format(region)
    fig.savefig(str(savename))


# In[20]:

get_ipython().magic('pinfo ax.legend')


# In[ ]:



