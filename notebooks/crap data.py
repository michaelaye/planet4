
# coding: utf-8

# In[4]:

from planet4 import io, reduction


# In[5]:

fname = ('/Users/klay6683/Dropbox/data/planet4/2015-10-11_'
         'planet_four_classifications_queryable_cleaned_seasons2and3.csv')


# In[6]:

data = reduction.read_csv_into_df(fname)


# In[7]:

image_ids = data.image_id.unique()


# In[8]:

df = data


# In[9]:

eps = 0.00001

f1 = (df.x.abs() < eps) & (df.y.abs() < eps)

f2 = ((df.radius_1 - 10).abs() < eps) & ((df.radius_2 - 10).abs() < eps)

f3 = (df.angle.abs() < eps) & ((df.distance - 10).abs() < eps)


# In[10]:

f = lambda d: ' and '.join(["{}=={}".format(k,v) for k,v in d.items()])


# In[11]:

f(dict(radius_1=10, radius_2=10))


# In[13]:

zeros = data[f1]


# In[30]:

zeros.marking.value_counts().sum()


# In[15]:

blotches = zeros[zeros.marking=='blotch']
fans = zeros[zeros.marking=='fan']


# In[16]:

blotches.query(f({'radius_1':10, 'radius_2':10})).shape[0]


# In[17]:

blotches[f2(blotches)].shape[0]


# In[18]:

fans.query(f(dict(angle=0))).shape[0]


# In[19]:

fans.angle.value_counts()


# In[20]:

fans.spread.value_counts()


# In[21]:

fans.distance.value_counts()


# In[22]:

4254 / data.shape[0]


# In[23]:

data.shape[0] - data[~((f1 & f2) | (f1 & f3))].shape[0]


# In[24]:

4254 + 5559


# In[32]:

data.shape[0] - 11569


# In[28]:

scanned = reduction.scan_for_incomplete_and_default(data)


# In[33]:

scanned.shape[0]


# In[34]:

data.shape[0] - scanned.shape[0]


# In[ ]:




# In[ ]:

eps = 0.00001
cont = []
for image_id in image_ids:
    blotches = data[(data.marking=='blotch') & (data.image_id==image_id)]
    at_zero = get_data_with_pos_zero(blotches)
    both = get_data_with_radii_10(at_zero)
    if at_zero.shape[0] > 0:
        cont.append(image_id)


# In[ ]:

at_zero = get_data_with_pos_zero(data)


# In[ ]:

at_zero_and_radius_10 = get_data_with_radii_10(at_zero)


# In[ ]:

bad_ids = at_zero_and_radius_10.image_id.unique()


# In[ ]:

bad_data = at_zero_and_radius_10


# In[ ]:

grouped = at_zero_and_radius_10.groupby(['image_id','classification_id','user_name'])


# In[ ]:

bad_id = bad_data[bad_data.image_id=='APF0001k80']


# In[ ]:

cols_of_interest = 'classification_id created_at user_name x y radius_1 radius_2 angle'.split()


# In[ ]:

bad_class_ids = grouped.size().order(ascending=False)


# In[ ]:

s = grouped.size().order(ascending=False)


# In[ ]:

s[s>15].sum()


# In[ ]:

s.sum()


# In[ ]:

data[data.marking=='blotch'].shape


# In[ ]:

s.sum() / 4152498.0


# In[ ]:

s


# In[ ]:

fans = data[data.marking=='fan']


# In[ ]:

fans_at_zero = get_data_with_pos_zero(fans)


# In[ ]:

fans_at_zero.shape


# In[ ]:

bad_fans = get_fans_with_default_data(fans_at_zero)


# In[ ]:

bad_fans.shape


# In[ ]:

grouped_fans = bad_fans.groupby(['image_id','classification_id','user_name'])


# In[ ]:

grouped_fans.size().order(ascending=False).to_csv('bad_fans.csv')


# In[ ]:

fans[(fans.x <0) & (fans.y < 0)].shape


# In[ ]:

data[(data.marking=='blotch') & (data.x < 0) & (data.y < 0)].shape


# In[ ]:

fans.shape


# In[ ]:

5764 / 3352349.0


# In[ ]:

fig, ax = subplots(nrows=2)
bad_fans.groupby('created_at').size().plot(style='*', title='Default value fans', ax=ax[0])
bad_data.groupby('created_at').size().plot(style='*', title='Default value blotches', ax=ax[1])


# In[ ]:

grouped_fans.size().order(ascending=False)


# In[ ]:

bad_data.angle.plot()


# In[ ]:

grouped_all = data['image_id user_name classification_id'.split()].groupby(['image_id','user_name'])


# In[ ]:

grouped_all.apply(lambda x: x.classification_id.nunique())


# In[ ]:




# In[ ]:

data[(data.image_id=='APF0000001') & (data.user_name =='JellyMonster')]


# In[ ]:

df = pd.DataFrame.from_dict(dict(image_id=[1,1,2,2,2], user_name=['a','a','c','d','d'],
                            classification_id=['ABC','ABC','GHI','JKL','MNO']))


# In[ ]:

df.groupby(['image_id','user_name']).classification_id.nunique()


# In[ ]:



