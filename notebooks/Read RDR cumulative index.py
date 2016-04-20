
# coding: utf-8

# In[ ]:

from planet4 import io


# In[ ]:

hirise_root = io.dropbox() / 'data/hirise'


# # Get column names first

# In[ ]:

lblfile = hirise_root / 'RDRCUMINDEX.LBL'


# In[ ]:

import pvl


# In[ ]:

label = pvl.load(lblfile)


# In[ ]:

table = label['RDR_INDEX_TABLE']


# In[ ]:

names = []
for item in table:
    second = item[1]
    if isinstance(second, pvl.PVLObject):
        names.append(second['NAME'])


# In[ ]:

names[:5]


# # now read index table file

# In[ ]:

indexfile = hirise_root / 'RDRCUMINDEX.TAB'


# In[ ]:

with indexfile.open() as f:
    data = f.readline()


# In[ ]:

data


# In[ ]:

df = pd.read_csv(indexfile, names=names).head()


# In[ ]:

df.columns


# In[ ]:



