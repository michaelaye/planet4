
# coding: utf-8

# # Task: Define status of Planet 4

# First import the `pandas` data table analyis library and check which version I'm using (as I'm constantly changing that to keep up-to-date.)

# In[ ]:

import planet4 as p4
import pandas as pd
from planet4 import io


# In[ ]:

db = io.DBManager()


# In[ ]:

df = pd.read_hdf(db_fname, 'df', stop=1e5)


# ## Image IDs
# For a simple first task, let's get a list of unique image ids, to know how many objects have been published.

# In[ ]:

img_ids = pd.read_hdf(db_fname, 'df', columns=['image_id'])


# In[ ]:

img_ids = df.image_id.unique()
print img_ids


# So, how many objects were online:

# In[ ]:

no_all = len(img_ids)
no_all


# ## Classification IDs
# Now we need to find out how often each image_id has been looked at. 
# For that we have the groupby functionality. 
# Specifically, because we want to know how many citizens have submitted a classification for each image_id, we need to group by the image_id and count the unique classification_ids within each image_id group. 
# 
# ### Uniqueness within Image_ID!
# We need to constrain for uniqueness because each classified object is included with the same classification_id and we don't want to count them more than once, because we are interested in the overall submission only for now.
# 
# In other words: Because the different fans, blobs and interesting things for one image_id have all been submitted with the same classification_id, I need to constrain to unique classification_ids, otherwise images with a lot of submitted items would appear 'more completed' just for having a lot of fan-content, and not for being analyzed by a lot of citizens, which is what we want.
# 
# First, I confirm that classification_ids indeed have more than 1 entry, i.e. when there was more than one object classified by a user:

# In[ ]:

df.groupby(df.classification_id, sort=False).size()


# Ok, that is the case.
# Now, group those classification_ids by the image_ids and save the grouping. Switch off sorting for speed, we want to sort by the counts later anyway.

# In[ ]:

grouping = df.classification_id.groupby(df.image_id, sort=False)


# Aggregate each group by finding the size of the unique list of classification_ids.

# In[ ]:

counts = grouping.agg(lambda x: x.unique().size)
counts


# Order the counts by value

# In[ ]:

counts = counts.order(ascending=False)
counts


# Note also that the length of this counts data series is 98096, exactly the number of unique image_ids.

# ## Percentages done.
# 
# By constraining the previous data series for the value it has (the counts) and look at the length of the remaining data, we can determine the status of the finished rate.

# In[ ]:

counts[counts >= 30].size


# That's pretty disappointing, but alas, the cold hard truth. 
# This means, taking all submitted years into account in the data, we have currently only the following percentage done:

# In[ ]:

counts[counts>= 30].size / float(no_all) * 100


# Wishing to see higher values, I was for some moments contemplating if one maybe has to sum up the different counts to be correct, but I don't think that's it.
# 
# The way I see it, one has to decide in what 'phase-space' one works to determine the status of Planet4.
# Either in the phase space of total subframes or in the total number of classifications. And I believe to determine the finished state of Planet4 it is sufficient and actually easier to focus on the available number of subframes and determine how often each of them has been looked at.
# 
# ## Separate for seasons
# The different seasons of our south polar observations are separated by several counts of the `thousands` digit in the `image_id` column of the original HiRISE image id, in P4 called image_name.

# In[ ]:

# str[5:7] is the 2-digit thousands count in, e.g., ESP_011234_0950, in this case 11.
df['thousands'] = df.image_name.str[5:7].astype('int')


# In[ ]:

thousands = df.thousands.value_counts().sort_index()
thousands


# As one can see, we have groups of [1..5, 11..13, 20..22].
# Let's add another season column to the dataframe, first filled with zeros.

# In[ ]:

df['season'] = 0


# For the first season, we actually don't need to look at the thousands counter, as the first 3 letters of the image_names started all with PSP in the first season (for 'P_rimary S_cience P_hase').
# Now let's set all rows with names starting with 'PSP' to season 1.

# In[ ]:

df.loc[:, 'season'][df.image_name.str.startswith('PSP')] = 1


# And for the later seasons, we actually need to group by the thousands counter:

# In[ ]:

df.loc[:, 'season'][(df.thousands > 10) & (df.thousands < 20)] = 2
df.loc[:, 'season'][df.thousands > 19] = 3


# So, for all seasons, how many rows to we have in the overall data:

# In[ ]:

no_all = df.season.value_counts()
no_all


# ### Percentages done
# Now I code a short function with the code I used above to create the counts of classification_ids per image_id. Note again the restriction to uniqueness of classification_ids.

# In[ ]:

def get_counts_per_classification_id(df, unique=True):
    grouping = df.classification_id.groupby(df.image_id, sort=False)
    # because I only grouped the classification_id column above, this function is only
    # applied to it. First, reduce to a unique list, and then save the size of that list.
    if unique:
        return grouping.agg(lambda x: x.unique().size)
    else:
        return grouping.size()


# In[ ]:

df.image_name.groupby(df.season).agg(lambda x:x.unique().size)


# In[ ]:

no_all = df.image_id.groupby(df.season).agg(lambda x: x.unique().size)
no_all


# In[ ]:

def done_per_season(season, limit, unique=True, in_percent=True):
    subdf = df[df.season == season]
    counts_per_classid = get_counts_per_classification_id(subdf, unique)
    no_done = counts_per_classid[counts_per_classid >= limit].size
    if in_percent:
        return 100.0 * no_done / no_all[season]
    else:
        return no_done


# In[ ]:

for season in [1,2,3]:
    print season
    print done_per_season(season, 30, in_percent=True)


# In the following code I not only check for the different years, but also the influence on the demanded limit of counts to define a subframe as 'finished'.
# 
# To collect the data I create an empty dataframe with an index ranging through the different limits I want to check (i.e. `range(30,101,10)`)

# In[ ]:

import sys
from collections import OrderedDict
results = pd.DataFrame(index=range(30,101,10))
for season in [1,2,3]:
    print season
    sys.stdout.flush() # to force a print out of the std buffer
    subdf = df[df.season == season]
    counts = get_counts_per_classification_id(subdf)
    values = OrderedDict()
    for limit in results.index:
        values[limit] = done_per_season(season, limit)
    results[season] = values.values()


# In[ ]:

np.round(results)


# # Problem ??
# ## Group by user_name instead of classification_id
# 
# I realised that user_ids should provide just the same access to the performed counts, because each classification_id should have exactly one user_id, as they are created when that user clicks on *Submit*, right? 
# At least that's how I understood it.
# 
# So imagine my surprise when I found out it isn't the same answer. And unfortunately it looks like we have to reduce our dataset even further by apparent multiple submissions of the same classification, but let's see.
# 
# First, create the respective function to determine counts via the user_name instead of classification_id after grouping for image_id.
# This first grouping by image_id is the essential step for the determination how often a particular image_id has been worked on, so that doesn't change.

# In[ ]:

def get_counts_per_user_name(df):
    grouping = df.user_name.groupby(df.image_id, sort=False)
    counts = grouping.agg(lambda x: x.unique().size)
#    counts = counts.order(ascending=False)
    return counts


# In[ ]:

counts_by_user = get_counts_per_user_name(df)
counts_by_user


# Compare that again to the output for classifying per classification_id:

# In[ ]:

counts_by_class = get_counts_per_classification_id(df)
counts_by_class


# So, _not_ the same result! Let's dig deeper.
# 
# ### The subframe known as jp7
# Focus on one image_id and study what is happening there. I first get a sub-table for the subframe 'jp7' and determine the user_names that worked on that subframe.
# 
# Then I loop over the names, filtering another sub-part of the table where the current user worked on jp7. 
# According to the hypothesis that a classification_id is created for a user at submisssion time and the idea that a user should not see an image twice, there should only be one classification_id in that sub-part.
# 
# I am testing that by checking if the unique list of classification_ids has a length $>1$. If it does, I print out the user_name.

# In[ ]:

jp7 = df[df.image_id == 'APF0000jp7']
unique_users = jp7.user_name.unique()
# having the list of users that worked on jp7
for user in unique_users:
    subdf = jp7[jp7.user_name == user]
    if len(subdf.classification_id.unique()) > 1:
        print user, len(subdf)


# Ok, so let's have a look at the data for the first user_name for the subframe jp7

# In[ ]:

jp7[jp7.user_name == 'not-logged-in-8d495c463aeffd67c08b2dfc1141f33b']


# First note that the creation time of these 2 different classifications is different, so it looks like this user has seen the jp7 subframe more than once.
# 
# But then when you scroll this html table to the right, you will notice that the submitted object has the exact same coordinates in both classifications? 
# How likely is it, that the user finds the exact same coordinates in less than 60 seconds?
# 
# So the question is, is this really a new classification and the user has done it twice? Or was the same thing submitted twice? Hopefully Meg knows the answer to that.

# ## Some instructive plots
# 
# ### Plot over required constraint
# 
# I found it instructive to look at how the status of finished data depends on the limit we put on the reached counts per image_id (i.e. subframe).
# 
# Also, how does it change when looking for unique user_names per image_id instead of unique classification_ids.

# In[ ]:

results[[2,3]].plot()
xlabel('Required number of analyses submitted to be considered "done".')
ylabel('Current percentage of dataset finished [%]')
title("Season 2 and 3 status, depending on definition of 'done'.")
savefig('Season2_3_status.png', dpi=200)


# In[ ]:

x = range(1,101)
per_class = []
per_user = []
for val in x:
    per_class.append(100 * counts_by_class[counts_by_class >= val].size/float(no_all))
    per_user.append(100 * counts_by_user[counts_by_user >= val].size/float(no_all))


# In[ ]:

plot(x,per_class)
plot(x, per_user)
xlabel('Counts constraint for _finished_ criterium')
ylabel('Current percent finished [%]')


# Ok, so not that big a deal until we require more than 80 classifications to be done.
# 
# ### How do the different existing user counts distribute
# 
# The method 'value_counts()' basically delivers a histogram on the counts_by_user data series.
# In other words, it shows how the frequency of classifications distribute over the dataset. It shows an to be expected peak close to 100, because that's what we are aiming now and the system does _today_ not anymore show a subframe that has been seen 100 times.
# 
# But it also shows quite some _waste_ in citizen power from all the counts that went for counts > 100.

# In[ ]:

counts_by_user.value_counts()


# In[ ]:

counts_by_user.value_counts().plot(style='*')


# In[ ]:

users_work = df.classification_id.groupby(df.user_name).agg(lambda x: x.unique().size)


# In[ ]:

users_work.order(ascending=False)[:10]


# In[ ]:

df[df.user_name=='gwyneth walker'].classification_id.value_counts()


# In[ ]:

import helper_functions as hf
reload(hf)


# In[ ]:

get_ipython().set_next_input("hf.classification_counts_for_user('Kitharode', df).hist");get_ipython().magic('pinfo hist')


# In[ ]:

hf.classification_counts_for_user('Paul Johnson', df)


# In[ ]:

np.isnan(df.marking)


# In[ ]:

df.marking


# In[ ]:

s = 'INVESTIGATION OF POLAR SEASONAL FAN DEPOSITS USING CROWDSOURCING'


# In[ ]:

s.title()


# In[ ]:



