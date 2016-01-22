
# coding: utf-8

# In[1]:

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
    d = {}
    d['image_name'] = img_name
    blotch_fname = root / '{}_blotches_latlons.csv'.format(img_name)
    d['n_blotches'] = len(pd.read_csv(str(blotch_fname)))
    fan_fname = root / '{}_fans_latlons.csv'.format(img_name)
    d['n_fans'] = len(pd.read_csv(str(fan_fname)))
    
    return d


# In[ ]:

get_final_markings_counts('ESP_020115_0985')


# In[ ]:

results = []
from IPython.display import display
from ipywidgets import IntProgress
t = IntProgress(min=0, max=len(region_data.Inca.season2)-1)
display(t)
for i, img_name in enumerate(region_data.Inca.season2):
    t.value = i
    try:
        results.append(get_final_markings_counts(img_name))
    except OSError:
        continue
season2 = pd.DataFrame(results).sort_values(by='image_name')


# In[ ]:

results = []
from IPython.display import display
from ipywidgets import IntProgress
t = IntProgress(min=0, max=len(region_data.Inca.season3)-1)
display(t)
for i, img_name in enumerate(region_data.Inca.season3):
    t.value = i
    try:
        results.append(get_final_markings_counts(img_name))
    except OSError:
        continue
season3 = pd.DataFrame(results).sort_values(by='image_name')


# In[ ]:

season2.info()


# In[ ]:

get_ipython().magic('matplotlib inline')


# In[ ]:

metadata = pd.read_csv("/Users/klay6683/Dropbox/data/planet4/P4_10-18-15_H_lat_lng.csv")


# In[ ]:

metadata


# In[ ]:

from hirise.hirise_tools import get_rdr_label, labels_root


# In[ ]:

import pvl
def get_nlines_from_label(labelfname):
    module = pvl.load(str(labelfname))
    return module['UNCOMPRESSED_FILE']['IMAGE']['LINE_SAMPLES']


# In[ ]:

p = labels_root()


# In[ ]:

metadata['labelpath'] = metadata.HiRISE_image.map(lambda x: p / (x + '_RED.LBL'))


# In[ ]:

metadata['nsamples'] = metadata.labelpath.map(get_nlines_from_label)


# In[ ]:

metadata.set_index('HiRISE_image', inplace=True)


# In[ ]:

season2.set_index('image_name', inplace=True)
season3.set_index('image_name', inplace=True)


# In[ ]:

season2 = season2.join(metadata['solar_longitude binning nsamples'.split()])


# In[ ]:

season3 = season3.join(metadata['solar_longitude binning nsamples'.split()])


# In[ ]:

season2.set_index('solar_longitude', inplace=True)


# In[ ]:

season3.set_index('solar_longitude', inplace=True)


# In[ ]:

season2['both'] = season2.n_blotches + season2.n_fans


# In[ ]:

season3['both'] = season3.n_blotches + season3.n_fans


# In[ ]:

season2['scaled'] = season2.both / season2.nsamples / season2.binning


# In[ ]:

season3['scaled'] = season3.both / season3.nsamples / season3.binning


# In[ ]:

get_ipython().magic('matplotlib notebook')
import seaborn as sns
sns.set()


# In[ ]:

season2.scaled.plot(style='*', ms=14, 
                    xlim=(season3.index.min(), season3.index.max()), label='season2')
season3.scaled.plot(style='*', color='red', label='season3', ms=14)
plt.legend(loc='best')
plt.ylabel('Scaled prevalence of markings')
plt.title("Number of markings in Inca City region,scaled for binning and image size.")
plt.savefig('/Users/klay6683/Desktop/inca_s23.pdf')


# In[ ]:

season2


# In[ ]:

season3


# In[ ]:

season3.scaled.plot()


# In[ ]:



