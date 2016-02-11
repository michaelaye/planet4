
# coding: utf-8

# In[ ]:

df1 = pd.DataFrame(np.arange(3), index='abc def ghi'.split(), columns=['somedata'])
df1


# In[ ]:

df2 = pd.DataFrame({'obsid':'abc def ghi'.split(),
                    'data':np.arange(3)*1000})
df2


# In[ ]:

df1.somedata/df2.data


# In[ ]:

df2.set_index('obsid', inplace=True)


# In[ ]:

df1


# In[ ]:

df2


# In[ ]:

df1.somedata/df2.data


# In[ ]:



