
# coding: utf-8

# In[ ]:

# setup
from planet4 import io
import tempfile

import planet4 as p4
from pathlib import Path
datapath = Path(p4.__path__[0]) / 'data'

import datetime as dt
from numpy.testing import assert_array_equal
import datetime as dt
import pytest


# In[ ]:

# test_P4DBName
name = '2015-10-21_planet_four_classifications.csv'
p4db = io.P4DBName(name)
assert p4db.name == name
assert p4db.p == Path(name)
assert p4db.parent == Path('.')
assert p4db.date == dt.datetime(2015, 10, 21)


# In[ ]:

# test_get_image_names_from_db
result = io.get_image_names_from_db(datapath / 'test_db.csv')
expected = np.array(['APF000012q', 'APF000012w'])
assert_array_equal(expected, result)


# In[ ]:

# test_get_current_database_fname
fname = io.get_current_database_fname()
assert isinstance(fname, Path)


# In[ ]:

# test_get_latest_file
l = ['2015-12-01_test.h5','2015-11-01_test.h5']
assert io.get_latest_file(l) == Path(l[0])
# empty list should raise NoFilesFoundError
with pytest.raises(p4.exceptions.NoFilesFoundError):
    io.get_latest_file([])


# In[ ]:

# test_PathManager_missing_id_
pm = io.PathManager('/tmp')
# testing exceptions
with pytest.raises(TypeError):
    assert pm.fanfile == Path('/tmp/ABC01234_fans.hdf')
    assert pm.blotchfile == Path('/tmp/ABC01234_blotches.hdf')
    assert pm.fnotchfile == Path('/tmp/ABC01234_fnotches.hdf')


# In[ ]:

# test_PathManager_proper_setup
pm = io.PathManager('/tmp')
pm.id_ = 'ABC01234'
assert pm.fanfile == Path('/tmp/ABC01234_fans.hdf')
assert pm.blotchfile == Path('/tmp/ABC01234_blotches.hdf')
assert pm.fnotchfile == Path('/tmp/ABC01234_fnotches.hdf')


# In[ ]:

# test_PathManager_suffix_.h5
pm = io.PathManager('/tmp', suffix='.h5')
pm.id_ = 'ABC01234'
assert pm.fanfile == Path('/tmp/ABC01234_fans.h5')
assert pm.blotchfile == Path('/tmp/ABC01234_blotches.h5')
assert pm.fnotchfile == Path('/tmp/ABC01234_fnotches.h5')


# In[ ]:

# test_PathManager_suffix_.csv
pm = io.PathManager('/tmp', suffix='.csv')
pm.id_ = 'ABC01234'
assert pm.fanfile == Path('/tmp/ABC01234_fans.csv')
assert pm.blotchfile == Path('/tmp/ABC01234_blotches.csv')
assert pm.fnotchfile == Path('/tmp/ABC01234_fnotches.csv')


# In[ ]:

# test_PathManager_setup_folders
tempdir = tempfile.TemporaryDirectory()
pm = io.PathManager(tempdir.name)
pm.setup_folders()
assert pm.fnotched_dir == Path(tempdir.name)
assert pm.output_dir_clustered == Path(tempdir.name) / 'just_clustering'


# In[ ]:

pm = io.PathManager()


# In[ ]:

pm.setup_folders()


# In[ ]:

pm.id_ = 'ABC'


# In[ ]:

pm.fanfile


# In[ ]:



