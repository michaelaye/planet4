
# coding: utf-8

# In[ ]:

from pathlib import Path
from planet4 import io, markings, plotting
# %matplotlib nbagg
import seaborn as sns
sns.set_style('white')


# In[ ]:

datapath = io.p4data() / 'season23_by_id'
fnotchedpath = io.p4data() / 'season23_by_id_fnotched_0.5'
pngpath = fnotchedpath.with_name(fnotchedpath.name + '_pipeline_check')
pngpath.mkdir(exist_ok=True)
dbpath = io.p4data() / '2015-11-02_planet_four_classifications_queryable_cleaned_seasons2and3.h5'


# In[ ]:

image_ids = [i.stem.split('_')[0] for i in fnotchedpath.glob('*.hdf')]


# In[ ]:

image_ids = np.unique(image_ids)


# In[ ]:

from mpl_toolkits.axes_grid1 import AxesGrid

def plot_pipeline_check(image_id):
    from planet4 import plotting, markings
    
    resman = plotting.ResultManager(image_id, datapath)
    p4id = markings.ImageID(image_id,database_fname = str(dbpath))
    fig, axes = plt.subplots(2,3, figsize=(12,6), sharex=True,
                             sharey=True)
    axes = axes.ravel()
    # plot tile, blotch and fan markings
    p4id.show_subframe(ax=axes[0])
    p4id.plot_fans(ax=axes[1])
    p4id.plot_blotches(ax=axes[2])
    
    # plot clustered blotches and fans
    try:
        p4id.plot_fans(ax=axes[4],
                       fans=resman.fans(unfnotched=True))
        p4id.plot_blotches(ax=axes[5], 
                           blotches=resman.blotches(unfnotched=True))
    except OSError as e:
        print(e)
        return False
    # plot fnotched results
    resman = plotting.ResultManager(image_id, fnotchedpath)
    p4id.plot_blotches(ax=axes[3], blotches=resman.blotches())
    p4id.plot_fans(ax=axes[3], fans=resman.fans())


    fig.subplots_adjust(left=None, top=None, bottom=None, right=None,
                        wspace=0.001, hspace=0.001)
    savepath = pngpath / (image_id + '_pipeline_check.pdf')
    fig.savefig(str(savepath), bbox_inches='tight')
    plt.close(fig)


# In[ ]:

image_ids[100]


# In[ ]:

for img_id in image_ids:
    plot_pipeline_check(img_id)


# In[ ]:

from ipyparallel import Client
c = Client()


# In[ ]:

lbview = c.load_balanced_view()


# In[ ]:

results = lbview.map_async(plot_pipeline_check, image_ids)


# In[ ]:

from iuvs.multitools import nb_progress_display


# In[ ]:

nb_progress_display(results, image_ids)


# In[ ]:

for res in results:
    res


# In[ ]:



