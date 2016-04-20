
# coding: utf-8

# In[1]:

from planet4 import region_data


# In[2]:

from hirise.hirise_tools import get_rdr_color_label


# In[3]:

regions = ['Giza', 'Ithaca', 'Manhattan2', 'Inca']


# In[4]:

seasons = ['season2', 'season3']


# In[5]:

from ipyparallel import Client
c = Client()
lbview = c.load_balanced_view()


# In[18]:

for region in regions:
    print(region)
    for season in seasons:
        print(season)
        reg = getattr(region_data, region)
        seas = getattr(reg, season)
        for img in seas:
            print(img)
            get_rdr_color_label(img)

