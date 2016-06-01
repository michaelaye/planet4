
# coding: utf-8

# In[ ]:

from planet4.clustering import DBScanner


# In[ ]:

dbname = '/Users/klay6683/data/planet4/2015-06-07_planet_four_classifications_queryable_cleaned_seasons2and3.h5'


# In[ ]:

from planet4 import markings, io

class ClusteringManager(object):
    def __init__(self, dbname, scope='hirise'):
        self.db = io.DBManager(dbname)
        self.dbname = dbname
        self.scope = scope
        self.confusion = []
        self.dbscanners = []
        self.clustered_fans = []
        self.clustered_blotches = []
        
    @property
    def n_clustered_fans(self):
        return len(self.clustered_fans)
    
    @property
    def n_clustered_blotches(self):
        return len(self.clustered_blotches)
    
    def dbscan_data(self, data):
        for kind in ['fan','blotch']:
            markings = data[data.marking==kind]
            dbscanner = DBScanner(markings, 
                                  kind, 
                                  scope=self.scope)
            self.confusion.append((self.data_id,
                                   kind,
                                   len(markings),
                                   dbscanner.n_reduced_data,
                                   dbscanner.n_rejected))
            if kind == 'fan':
                self.clustered_fans.extend(dbscanner.reduced_data)
            else:
                self.clustered_blotches.extend(dbscanner.reduced_data)

    def dbscan_image_id(self, image_id):
        self.data_id = image_id
        self.p4id = markings.ImageID(image_id, self.dbname)
        self.dbscan_data(self.p4id.data)
        
    def dbscan_image_name(self, image_name):
        data = self.db.get_image_name_markings(image_name)
        self.data_id = image_name
        self.dbscan_data(data)
        
    def dbscan_all(self):
        image_names = self.db.image_names
        for i, image_name in enumerate(image_names):
            print('{:.1f}'.format(100*i/len(image_names)))
            data = self.db.get_image_name_markings(image_name)
            self.data_id = image_name
            self.dbscan_data(data)


# In[ ]:

cm = ClusteringManager(dbname)


# In[ ]:

cm.dbscan_image_name('ESP_011544_0985')


# In[ ]:

from numpy.linalg import norm

n_close = 0
for blotch in cm.clustered_blotches:
    for fan in cm.clustered_fans:
        delta = blotch.center - (fan.base+fan.midpoint)
        if norm(delta) < 10 :
           n_close += 1 
    


# In[ ]:

n_close


# In[ ]:

cm.n_clustered_blotches


# In[ ]:

cm.n_clustered_fans


# In[ ]:

confusion_data = pd.DataFrame(cm.confusion, columns=['image_name', 'kind', 'n_markings',
                                    'n_cluster_members', 'n_rejected'])


# In[ ]:

confusion_data.to_csv('/Users/klay6683/Dropbox/DDocuments/planet4/confusion_data.csv')


# In[ ]:

from numpy.linalg import norm

for blotch in reduced_blotches:
    print(blotch.center)
    for fan in reduced_fans:
        delta = blotch.center - (fan.base+fan.midpoint)
        print(norm(delta))

