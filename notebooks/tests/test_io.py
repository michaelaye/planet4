
# coding: utf-8

# In[1]:

# setup
from planet4 import io
import tempfile
import numpy as np
import planet4 as p4
from pathlib import Path
import datetime as dt
from numpy.testing import assert_array_equal
import datetime as dt
import pytest
import pkg_resources as pr

datapath = Path(pr.resource_filename('planet4', 'data'))


# In[2]:

# test_P4DBName
name = '2015-10-21_planet_four_classifications.csv'
p4db = io.P4DBName(name)
assert p4db.name == name
assert p4db.p == Path(name)
assert p4db.parent == Path('.')
assert p4db.date == dt.datetime(2015, 10, 21)


# In[3]:

# test_get_image_names_from_db
result = io.get_image_names_from_db(datapath / 'test_db.csv')
expected = np.array(['APF000012q', 'APF000012w'])
assert_array_equal(expected, result)


# In[4]:

# test_get_latest_file
l = ['2015-12-01_test.h5', '2015-11-01_test.h5']
assert io.get_latest_file(l) == Path(l[0])
# empty list should raise NoFilesFoundError
with pytest.raises(p4.exceptions.NoFilesFoundError):
    io.get_latest_file([])


# In[5]:

# test_PathManager_missing_id_
pm = io.PathManager('/tmp')
# testing exceptions
with pytest.raises(TypeError):
    assert pm.fanfile == Path('/tmp/ABC0001234_fans.hdf')
    assert pm.blotchfile == Path('/tmp/ABC0001234_blotches.hdf')
    assert pm.fnotchfile == Path('/tmp/ABC0001234_fnotches.hdf')


# In[9]:

# test_PathManager_default_setup
pm = io.PathManager('/tmp')
pm.id_ = 'ABC0001234'
assert pm.fanfile == Path('/tmp/ABC0001234_fans.csv')
assert pm.blotchfile == Path('/tmp/ABC0001234_blotches.csv')
assert pm.fnotchfile == Path('/tmp/ABC0001234_fnotches.csv')


# In[10]:

# test_PathManager_different_suffix
pm = io.PathManager('/tmp', suffix='.hdf')
pm.id_ = 'ABC0001234'
assert pm.fanfile == Path('/tmp/ABC0001234_fans.hdf')
assert pm.blotchfile == Path('/tmp/ABC0001234_blotches.hdf')
assert pm.fnotchfile == Path('/tmp/ABC0001234_fnotches.hdf')


# In[11]:

# test_PathManager_suffix_.csv
pm = io.PathManager('/tmp', suffix='.csv')
pm.id_ = 'ABC0001234'
assert pm.fanfile == Path('/tmp/ABC0001234_fans.csv')
assert pm.blotchfile == Path('/tmp/ABC0001234_blotches.csv')
assert pm.fnotchfile == Path('/tmp/ABC0001234_fnotches.csv')


# In[12]:

# test_PathManager_setup_folders
tempdir = tempfile.TemporaryDirectory()
pm = io.PathManager(tempdir.name)
pm.setup_folders()
assert pm.output_dir == Path(tempdir.name)
assert pm.output_dir_clustered == Path(tempdir.name) / 'just_clustering'
