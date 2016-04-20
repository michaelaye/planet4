
# coding: utf-8

# In[3]:

from planet4 import region_data, io
import pvl
from hirise import hirise_tools as ht
import gdal


# In[4]:

regions = ['Inca', 'Ithaca', 'Giza', 'Manhattan2']


# In[5]:

seasons = ['season2', 'season3']


# In[6]:

def get_fraction_of_black_pixels(savepath):
    ds = gdal.Open(str(savepath))
    data = ds.ReadAsArray()
    fractions = []
    for band in data:
        nonzeros = band.nonzero()[0]
        fractions.append((band.size - nonzeros.size)/band.size)
    return np.array(fractions).mean()


# In[7]:

def read_metadata(obsid):
    prodid = ht.PRODUCT_ID(obsid)
    prodid.kind = 'COLOR'
    labelpath = ht.labels_root() / prodid.label_fname
    label = ht.HiRISE_Label(labelpath)
    savedir = ht.hirise_dropbox() / 'browse'
    savepath = savedir / prodid.browse_path.name
    if not savepath.exists():
        ht.download_product(prodid.browse_path, savedir)
    black_fraction = get_fraction_of_black_pixels(savepath)
    all_area = label.line_samples*label.lines * label.map_scale**2
    real_area = (1-black_fraction)*all_area
    d = dict(obsid=obsid, path=labelpath, binning=label.binning_color,
             l_s=label.l_s, line_samples=label.line_samples,
             lines=label.lines, map_scale=label.map_scale, invalids=black_fraction,
             real_area=real_area)
    
    return d


# In[8]:

for region in regions:
    rea = getattr(region_data, region)
    for season in seasons:
        sea = getattr(rea, season)
        metadata = []
        for img in sea:
            metadata.append(read_metadata(img))
        if region.endswith('2'):
            region = region[:-1]
        name = "{}_{}_metadata.csv".format(region.lower(), season)
        fname = io.analysis_folder() / name
        print(fname)
        pd.DataFrame(metadata).to_csv(str(fname), index=False)


# In[ ]:



