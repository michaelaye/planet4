
# coding: utf-8

# In[1]:

from planet4 import io


# In[2]:

from planet4.region_data import Inca


# In[3]:

db = io.DBManager()


# In[4]:

db.dbname


# In[5]:

data = db.get_all()


# In[6]:

db.dbname


# In[30]:

c_id = '50ef419195e6e40eac000001'
sub = data[data.classification_id==c_id]
sub.shape


# In[31]:

sub.T


# In[32]:

c_id = '50ef41ea95e6e42e89000001'
sub2 = data[data.classification_id==c_id]
sub2.shape


# In[33]:

sub2.T


# In[8]:

image_ids = data.image_id.unique()


# In[9]:

df = data[data.image_id==image_ids[0]]


# In[34]:

df = data[data.image_id=='APF00004y6']


# In[35]:

df.info()


# In[36]:

g = df.groupby(['user_name', 'classification_id'], sort=False)


# In[19]:

g.size()


# In[56]:

def show_duplicates(df):
    f = lambda x: len(x.classification_id.unique())
    return df.groupby(['user_name']).apply(f).sort_values(ascending=False)
    


# In[58]:

show_duplicates(df).head()


# In[61]:

def process_user_group(g):
    c_id = g.sort_values(by='created_at').classification_id.iloc[0]
    return g[g.classification_id == c_id]


# In[66]:

kyle = df[df.user_name=='Kyle Butcher']


# In[71]:

kyle.created_at


# In[87]:

df2 = data[data.image_id=='APF00003hp']


# In[88]:

show_duplicates(df2).head()


# In[107]:

user2 = df2[df2.user_name=='Kitharode']


# In[111]:

df2[df2.user_name=='Kitharode']


# First we find the earliest timestamp, each classification_id has in principle its own timestamp, like Kitharode's data above.
# But for that not-logged-in user, it has two classification_ids with the same time-stamp:

# In[112]:

user2 = df2[df2.user_name=='not-logged-in-073dee28bbc9c250d9dc02cb99f4ef93']


# In[114]:

user2[user2.created_at==user2.created_at.min()].classification_id.unique()


# Note that above is filtered for data at minimum time!

# but simply doing another minimum(), like i did in previous version of filtering for this, should always work

# In[116]:

user2[user2.created_at==user2.created_at.min()].classification_id.min()


# For explanation: this is the earliest AND smallest classification_id for this image_id.

# In[140]:

def process_image_name(data):

    def process_user_group(g):
        c_id = g[g.created_at==g.created_at.min()].classification_id.min()
        return g[g.classification_id == c_id]

    data = data.groupby(['image_id', 'user_name'], sort=False).apply(
        process_user_group).reset_index(drop=True)
    return data
#     fname = 'temp_' + image_name + '.h5'
#     data.to_hdf(fname, 'df')


# In[141]:

imgname_data = data[data.image_name=='ESP_011737_0980']


# In[142]:

tmp = process_image_name(imgname_data)


# In[143]:

tmp


# In[ ]:

filtered_data = df.groupby(['user_name']).apply(process_user_group)


# In[ ]:

from planet4 import helper_functions as hf


# In[ ]:

def process_image_id(image_id):
    df = data[data.image_id==image_id]
    n_class = df.classification_id.unique().size
    results = hf.classification_counts_per_user(df).value_counts()
    if not any(results.index>1):
        return n_class, n_class
    else:
        n_class_real = results[1]
        for index in results.index[results.index>1]:
            n_class_real += results[index]
        return (n_class_real, n_class)


# In[ ]:

real_class_percents = []
for image_id in image_ids:
    real_class_percents.append(process_image_id(image_id))


# In[ ]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# In[ ]:

s = pd.DataFrame(real_class_percents, index=image_ids, columns=['real_n_class', 'expected_n_class'])


# In[ ]:

s.head()


# In[ ]:

s.plot()


# In[ ]:

import seaborn as sns
sns.set_context('talk')


# In[ ]:

s.head()


# In[ ]:

s = s.assign(fraction=s.real_n_class/s.expected_n_class)


# In[ ]:

s.fraction.plot(style='.',title='Inca City, season1, fraction of good classifications')


# In[ ]:

s.describe()


# In[ ]:

df = data[data.image_id=='APF0000zea']
n_class= df.classification_id.unique().size
n_class
results = hf.classification_counts_per_user(df).value_counts()
results
results.index>1
n_class_real = results[1]
for index in results.index[results.index>1]:
    n_class_real += results[index]
n_class_real


# # Update

# In[1]:

from planet4 import io


# In[2]:

from planet4 import reduction


# In[21]:

db = io.DBManager("/Users/klay6683/data/planet4/2016-04-17_planet_four_classifications_queryable.h5")


# In[22]:

db.dbname


# In[24]:

data = db.get_all()


# In[118]:

data = data[data.image_id=='APF00003hp']


# In[26]:

from planet4 import helper_functions as hf


# In[27]:

data.info()


# In[ ]:




# In[ ]:

def pug1(g):
    c_id = g.sort_values(by='classification_id').classification_id.iloc[0]
    return g[g.classification_id == c_id]


# In[ ]:

def pug1b(g):
    c_id = g.sort_values(by='classification_id').classification_id.iloc[0]
    return c_id


# In[ ]:

def pug2(g):
    c_id = g.classification_id.min()
    return g[g.classification_id == c_id]


# In[ ]:

def pug2b(g):
    c_id = g.classification_id.min()
    return c_id


# In[ ]:

usergroup = data.groupby(['user_name'], sort=False)


# In[ ]:

get_ipython().magic('timeit usergroup.apply(pug1).reset_index(drop=True)')


# In[ ]:

get_ipython().magic('timeit usergroup.apply(pug2).reset_index(drop=True)')


# In[ ]:

get_ipython().magic('timeit data[data.classification_id.isin(usergroup.classification_id.min())]')


# In[ ]:

v1 = usergroup.apply(pug1).reset_index(drop=True).sort_values(by=['classification_id'])


# In[ ]:

v2 = usergroup.apply(pug2).reset_index(drop=True).sort_values(by=['classification_id'])


# In[ ]:

v3 = data[data.classification_id.isin(usergroup.classification_id.min())]


# In[ ]:

(v1.dropna() == v3.dropna()).all()


# In[ ]:

v3[v3.classification_id=='50ef44b995e6e42d8c000001']


# In[ ]:

(usergroup.apply(pug1).dropna() == usergroup.apply(pug2).dropna()).all()


# In[ ]:

v1.info()


# In[ ]:

(v1.dropna()==v2.dropna()).all()


# In[ ]:

v2.head()


# In[20]:

data.groupby(['user_name','classification_id']).apply(lambda x: len(x.classification_id.unique())).sort_values(ascending=False).min()


# In[ ]:

v3 = data[data.classification_id.isin(data.groupby('user_name').classification_id.max())].sort_values(by='classification_id')


# In[ ]:

(v1.classification_id.sort_values() == v2.classification_id.sort_values()).all()


# In[ ]:

(g.apply(pug1).reset_index(drop=True) == g.apply(pug2).reset_index(drop=True)).all()


# In[ ]:

get_ipython().magic('timeit g.apply(pug2b)')


# In[ ]:

img_ids = ['APF00003hp']
users = ['not-logged-in-073dee28bbc9c250d9dc02cb99f4ef93']

for img_id, user in zip(img_ids, users):
    print("image_id: ", img_id)
    print("User: ", user)
    data = db.get_image_id_markings(img_id)
    print("Before filtering classification_id created_at times:")
    print(data[data.user_name==user].created_at.unique())
    print("Classification_ids:")
    print(data[data.user_name==user].classification_id.unique())

    g = data.groupby(['user_name'])
    res = g.apply(process_user_group).reset_index(drop=True)
    print("After filtering:")
    print(res[res.user_name==user].created_at.unique())
    print(res[res.user_name==user].classification_id.unique())
    print()


# In[ ]:

fname = '/Users/klay6683/data/planet4/2015-10-11_planet_four_classifications_queryable.h5'
db = io.DBManager()


# In[ ]:

db.dbname


# In[ ]:

data = pd.read_hdf(db.dbname, 'df',
                    where="classification_id=='50ef44b795e6e42cd2000001'")
data


# In[ ]:

df = pd.read_hdf(db.dbname, 'df')


# In[ ]:

df[df.classification_id=='50ef44b795e6e42cd2000001']


# In[ ]:

df[df.classification_id=='50ef44b995e6e42d8c000001']


# In[ ]:

df[df.classification_id=='50ee0e5694b9d564a90000b5']


# In[ ]:

db.dbname


# In[ ]:

df.classification_id = df.classification_id.astype('str')


# In[ ]:

df.to_hdf('/Users/klay6683/data/planet4/2015-10-11_planet_four_classifications_queryable_cleaned.h5',
         'df', format='t', data_columns=reduction.data_columns)


# In[ ]:

pd.read_hdf('/Users/klay6683/data/planet4/2015-10-11_planet_four_classifications_queryable_cleaned.h5',
           'df', where="classification_id=='50ef44b795e6e42cd2000001'")


# In[ ]:

fname = '/Users/klay6683/data/planet4/2015-10-11_planet_four_classifications_queryable.h5'


# In[ ]:

reduction.remove_duplicates_from_file(fname)


# In[ ]:

data = pd.read_hdf('testing.h5', 'df',
                    where="classification_id=='50ef44b995e6e42d8c000001'")
data


# In[ ]:

data2 = db.get_class_id_data('50ef44b995e6e42d8c000001')


# In[ ]:

s = pd.Series(list('abc'))


# In[ ]:

pd.DataFrame(s)


# In[ ]:

from planet4 import io


# In[ ]:

db=io.DBManager()


# In[ ]:

import time


# In[ ]:

imgnames = db.season2and3_image_names


# In[ ]:

where = "image_name in {}".format(imgnames.values.tolist())


# In[ ]:

where


# In[ ]:

import time
t0 = time.time()
season23 = pd.read_hdf(db.dbname, 'df', where=where)
t1 = time.time()
print("time: ", t1 - t0)


# In[ ]:



