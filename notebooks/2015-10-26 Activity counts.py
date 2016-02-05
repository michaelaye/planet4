
# coding: utf-8

# In[ ]:

from planet4 import region_data, io
from pathlib import Path


# In[ ]:

root = io.analysis_folder() / 'inca_s23_0.5cut_applied/'


# In[ ]:

fan_fnames = list(root.glob("*_fans_latlons.csv"))


# In[ ]:

blotch_fnames = list(root.glob("*_blotches_latlons.csv"))


# In[ ]:

from planet4 import markings


# In[ ]:

def get_marking(line, cut=0.8):
    data = []
    for marking in ['fan_','blotch_']:
        m = line[line.index.str.startswith(marking)]
        data.append(m.rename_axis(lambda x: x[x.index('_')+1:]))
    fnotch = markings.Fnotch(line.fnotch_value, data[0], data[1])
    return fnotch.get_marking(cut)


# In[ ]:

def get_final_markings_counts(img_name, cut=0.5):
    # writing in dictionary here b/c later I convert it to pd.DataFrame
    # for which a dictionary is a natural input format
    d = {}
    d['obsid'] = img_name
    blotch_fname = root / '{}_blotches_latlons.csv'.format(img_name)
    d['n_blotches'] = len(pd.read_csv(str(blotch_fname)))
    fan_fname = root / '{}_fans_latlons.csv'.format(img_name)
    d['n_fans'] = len(pd.read_csv(str(fan_fname)))
    
    return d


# In[ ]:

get_final_markings_counts('ESP_020115_0985')


# In[ ]:

from nbtools import ListProgressBar


# In[ ]:

results = []
progbar = ListProgressBar(region_data.Inca.season2)
for img_name in region_data.Inca.season2:
    progbar.value = img_name
    try:
        results.append(get_final_markings_counts(img_name))
    except OSError:
        continue
season2 = pd.DataFrame(results).sort_values(by='obsid')


# In[ ]:

results = []
progbar = ListProgressBar(region_data.Inca.season3)
for img_name in region_data.Inca.season3:
    progbar.value = img_name
    try:
        results.append(get_final_markings_counts(img_name))
    except OSError:
        continue
season3 = pd.DataFrame(results).sort_values(by='obsid')


# In[ ]:

season2.head()


# In[ ]:

season2.head()


# In[ ]:

season2_meta = pd.read_csv(io.analysis_folder() / 'inca_season2_metadata.csv')
# dropping the label path here as it's not required
# if it is, delete this line.
season2_meta.drop('path', axis=1, inplace=True)


# In[ ]:

season2_meta.head()


# In[ ]:

season2 = season2.merge(season2_meta, on='obsid')


# In[ ]:

season2.head()


# In[ ]:

path = io.analysis_folder() / 'inca_season3_metadata.csv'
season3_meta = pd.read_csv(path)


# In[ ]:

season3 = season3.merge(season3_meta, on='obsid')


# In[ ]:

season2.set_index('l_s', inplace=True)

season3.set_index('l_s', inplace=True)


# In[ ]:

season2['both'] = season2.n_blotches + season2.n_fans

season3['both'] = season3.n_blotches + season3.n_fans


# In[ ]:

season2['scaled'] = season2.both / season2.line_samples / season2.binning

season3['scaled'] = season3.both / season3.line_samples / season3.binning


# In[ ]:

get_ipython().magic('matplotlib notebook')


# In[ ]:

season2.scaled.plot(style='*', ms=14, 
                    xlim=(season3.index.min(), season3.index.max()), label='season2')
season3.scaled.plot(style='*', color='red', label='season3', ms=14)
plt.legend(loc='best')
plt.ylabel('Scaled prevalence of markings')
plt.title("Number of markings in Inca City region,scaled for binning and image size.")
plt.savefig('/Users/klay6683/Desktop/inca_s23.pdf')

