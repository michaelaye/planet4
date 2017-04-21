# coding: utf-8

# setup
from planet4 import io
import numpy as np
import planet4 as p4
from pathlib import Path
import datetime as dt
from numpy.testing import assert_array_equal
import pytest
import pkg_resources as pr

datapath = Path(pr.resource_filename('planet4', 'data'))


def test_P4DBName():
    name = '2015-10-21_planet_four_classifications.csv'
    p4db = io.P4DBName(name)
    assert p4db.name == name
    assert p4db.p == Path(name)
    assert p4db.parent == Path('.')
    assert p4db.date == dt.datetime(2015, 10, 21)


def test_get_image_names_from_db():
    result = io.get_image_names_from_db(datapath / 'test_db.csv')
    expected = np.array(['APF000012q', 'APF000012w'])
    assert_array_equal(expected, result)


def test_get_latest_file():
    l = ['2015-12-01_test.h5', '2015-11-01_test.h5']
    assert io.get_latest_file(l) == Path(l[0])
    # empty list should raise NoFilesFoundError
    with pytest.raises(p4.exceptions.NoFilesFoundError):
        io.get_latest_file([])


class TestPathManager:
    @pytest.fixture
    def data_root(self, monkeypatch):
        monkeypatch.setattr(io, 'data_root', Path('/abc/db'))

    @pytest.fixture(autouse=True)
    def pm(self, data_root):
        return io.PathManager('007')

    def test_no_datapath_given(self, pm):
        assert pm.datapath == Path('/abc/db/clustering')

    def test_suffix(self, pm):
        assert pm.suffix == '.csv'

    def test_relative_datapath_given(self, data_root):
        pm = io.PathManager('007', datapath='testing')
        assert pm.datapath == Path('/abc/db/testing')

    def test_absolute_datapath_given(self, data_root):
        pm = io.PathManager('007', datapath='/abc/db/extra')
        assert pm.datapath == Path('/abc/db/extra')

    def test_datapath_with_obsid(self, data_root):
        # no change despite giving an obsid:
        obsid = 'PSP_003092_0985'
        pm = io.PathManager('007', obsid=obsid)
        # still this, as obsid only added at each specific case
        assert pm.datapath == Path('/abc/db/clustering')
        assert pm.obsid == obsid

    def test_blotchfile(self, pm):
        assert pm.blotchfile ==\
            Path('/abc/db/clustering/APF0000007/APF0000007_blotches.csv')

    def test_fanfile(self, pm):
        assert pm.fanfile ==\
            Path('/abc/db/clustering/APF0000007/APF0000007_fans.csv')

    # def test_get_obsid_paths(self):
    #     pm = io.PathManager(obsid=obsid)


# def test_PathManager_default_setup():
#     pm = io.PathManager('ABC0001234')
#     assert pm.fanfile == Path('/tmp/ABC0001234_fans.csv')
#     assert pm.blotchfile == Path('/tmp/ABC0001234_blotches.csv')
#     assert pm.fnotchfile == Path('/tmp/ABC0001234_fnotches.csv')
#
#
# def test_PathManager_different_suffix():
#     pm = io.PathManager('/tmp', suffix='.hdf')
#     pm.id_ = 'ABC0001234'
#     assert pm.fanfile == Path('/tmp/ABC0001234_fans.hdf')
#     assert pm.blotchfile == Path('/tmp/ABC0001234_blotches.hdf')
#     assert pm.fnotchfile == Path('/tmp/ABC0001234_fnotches.hdf')
#
#
# def test_PathManager_suffix_csv():
#     pm = io.PathManager('/tmp', suffix='.csv')
#     pm.id_ = 'ABC0001234'
#     assert pm.fanfile == Path('/tmp/ABC0001234_fans.csv')
#     assert pm.blotchfile == Path('/tmp/ABC0001234_blotches.csv')
#     assert pm.fnotchfile == Path('/tmp/ABC0001234_fnotches.csv')
#
#
# def test_PathManager_setup_folders():
#     tempdir = tempfile.TemporaryDirectory()
#     pm = io.PathManager(tempdir.name)
#     pm.setup_folders()
#     assert pm.output_dir == Path(tempdir.name)
#     assert pm.output_dir_clustered == Path(tempdir.name) / 'just_clustering'
