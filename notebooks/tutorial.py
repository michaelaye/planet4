
# coding: utf-8

# In[ ]:

from planet4 import get_data, markings
reload(markings)


# In[ ]:

tuts = get_data.get_latest_tutorial_data()


# In[ ]:

tuts.info()


# In[ ]:

users = tuts.user_name.unique()


# In[ ]:

import logging


# In[ ]:

log = logging.getLogger()


# In[ ]:

log.setLevel(logging.INFO)


# In[ ]:

user = users[randint(0,len(users))]
userdata = tuts[tuts.user_name==user]
idobj = markings.P4_ImgID(userdata.image_id.iloc[0], data=userdata)
idobj.plot_fans()


# In[ ]:



