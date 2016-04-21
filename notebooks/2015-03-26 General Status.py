
# coding: utf-8

# In[ ]:

from planet4 import region_data, io, helper_functions


# # per region

# In[ ]:

def process_region(args):
    from planet4 import region_data, io, helper_functions
    regionclass, season = args
    try:
        imagenames = getattr(regionclass, season)
    except AttributeError:
        return "No {} data for region {} in PlanetFour.".format(season, regionclass)
    print(imagenames)
    df = io.get_list_of_image_names_data(imagenames)
    return "Status for {}, {}: {}".format(regionclass,
                                          season,
                                          helper_functions.get_status_per_classifications(df))


# In[ ]:

processlist = []
for season in ['season'+str(i) for i in range(1,5)]:
    for region in [region_data.Manhattan]:
        processlist.append((region, season))


# In[ ]:

from ipyparallel import Client
c = Client()
lbview = c.load_balanced_view()


# In[ ]:

import sys
for region in region_data.regions:
    print(region)
    sys.stdout.flush()
    processlist = []
    for season in ['season'+str(i) for i in range(1,5)]:
        processlist.append((region, season))
    res = lbview.map_async(process_region, processlist)
    for result in res.result:
        print(result)


# # per image_name (= hirise obs_id)

# In[ ]:

from planet4 import io


# In[ ]:

# image_names = io.get_all_image_names()
# image_names = image_names.image_name.unique()
# with open('current_image_names.pkl', 'wb') as f:
#     pickle.dump(image_names, f)


# In[ ]:

image_names = io.get_image_names_from_db(io.get_current_database_fname())


# In[ ]:

from planet4 import helper_functions as hf


# In[ ]:

def process_image_name(image_name):
    from planet4 import helper_functions as hf
    from planet4 import io
    df = io.get_image_name_data(image_name)
    status = hf.get_status_per_classifications(df)
    return image_name, status


# In[ ]:

def process_image_name(image_name):
    from planet4 import helper_functions as hf
    from planet4 import io
    df = io.get_image_name_data(image_name)
    no_done = hf.get_no_tiles_done(df)
    return {'image_name': image_name, 'no_done': no_done}


# In[ ]:

from ipyparallel import Client
c = Client()
lbview = c.load_balanced_view()


# In[ ]:

results = lbview.map_async(process_image_name, image_names)


# In[ ]:

for result in results:
    print(result)


# In[ ]:

df = pd.DataFrame(results.result)
df = df.set_index('image_name')
df


# In[ ]:

df.no_done.sum()


# In[ ]:

get_ipython().magic('matplotlib nbagg')


# In[ ]:

df = df.drop('tutorial')


# In[ ]:

df.plot(kind='hist',bins=30)


# In[ ]:

df = pd.read_hdf('image_name_status.h5','df')


# In[ ]:

df.head()


# # Figure for HiRISE meeting

# In[ ]:

import helper_functions as hf
reload(hf)

pd.__version__

df = hf.get_current_cleaned()

users_work = hf.classification_counts_per_user(df)
topten = users_work.order(ascending=False)[:10]

topten

hf.classification_counts_for_user('Kitharode',df)

for user in topten.index:
    print user
    print df[df.user_name==user].marking.value_counts()

get_ipython().set_next_input('df.marking.value_counts().plot');get_ipython().magic('pinfo plot')

s = df.marking.value_counts()

s.plot(kind='bar')
title('Marking stats')

savefig("marking_stats.png",dpi=200)

df.columns

no_nones = df[df.marking != 'None']

hf.get_top_ten_users(no_nones)

topten = hf.get_top_ten_users(df)

topten.plot(kind='bar')
title("Top ten citizens, submissions")
savefig('top_ten_submitters.png',dpi=200)

