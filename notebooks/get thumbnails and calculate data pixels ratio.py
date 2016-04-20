
# coding: utf-8

# In[31]:

import gdal

from hirise import hirise_tools as ht


# In[89]:

from planet4 import region_data


# In[90]:

region = region_data.Inca


# In[91]:

season = region.season2


# In[93]:

dir = ht.hirise_dropbox() / 'browse'


# In[183]:

def get_fraction_of_black_pixels(savepath):
    ds = gdal.Open(str(savepath))
    data = ds.ReadAsArray()
    fractions = []
    for band in data:
        nonzeros = band.nonzero()[0]
        fractions.append((band.size - nonzeros.size)/band.size)
    return np.array(fractions).mean()


# In[209]:

browse_fractions = []
for img in season:
    print(img)
    prodid = ht.PRODUCT_ID(img)
    prodid.kind = 'COLOR'
    savepath = dir / prodid.browse_path.name
#     ht.download_product(prodid.thumbnail_path, dir)
    browse_fractions.append(get_fraction_of_black_pixels(savepath))


# In[210]:

df = pd.DataFrame({'obsid':season, 'thumbnails':thumb_fractions,
                   'browse':browse_fractions}).set_index('obsid')


# In[211]:

get_ipython().magic('matplotlib inline')


# In[212]:

df.plot(rot=75, title='Inca, season2, fraction of invalid pixels in map-projected image.')


# In[ ]:



