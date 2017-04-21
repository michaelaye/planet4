# setup
import tempfile
from pathlib import Path

import pandas as pd
import pkg_resources as pr
import pytest

from planet4 import clustering

with pr.resource_stream('planet4', 'data/test_db.csv') as f:
    data = pd.read_csv(f)

# import warnings
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore",
#                             category=DeprecationWarning)
#


@pytest.fixture(autouse=True, scope='module')
def tdir():
    _tdir = tempfile.mkdtemp()
    yield Path(_tdir.name)
    # teardown
    _tdir.cleanup()


imid1 = 'APF000012w'
imid2 = 'APF000012q'

imid1data = data[data.image_id == imid1]
imid2data = data[data.image_id == imid2]

fans1 = imid1data[imid1data.marking == 'fan']
blotches1 = imid1data[imid1data.marking == 'blotch']
fans2 = imid2data[imid2data.marking == 'fan']
blotches2 = imid2data[imid2data.marking == 'blotch']

# basic clustering manager
cm = clustering.ClusteringManager(dbname='ignore', output_dir=tdir)


def test_calc_fnotch():
    actual = clustering.calc_fnotch(4, 4)
    assert actual == 0.5
    actual = clustering.calc_fnotch(4, 0)
    assert actual == 1
    actual = clustering.calc_fnotch(0, 4)
    assert actual == 0


def test_dbscan_xy_blotch():

    # using only x and y (or image_x,image_y)
    coords = ['image_x', 'image_y']
    X = blotches1[coords].values
    dbscanner = clustering.DBScanner(X, min_samples=2)

    assert dbscanner.n_clusters == 26
    assert dbscanner.n_rejected == 25


def test_dbscan_xy_fan():
    # using only x and y (or image_x,image_y)
    coords = ['image_x', 'image_y']
    X = fans1[coords].values
    dbscanner = clustering.DBScanner(X, min_samples=2)

    assert dbscanner.n_clusters == 7
    assert dbscanner.n_rejected == 11


def test_dbscan_xy_angle_blotch():
    coords = ['image_x', 'image_y', 'angle']
    X = blotches1[coords].values
    dbscanner = clustering.DBScanner(X, min_samples=2)

    assert dbscanner.n_clusters == 35
    assert dbscanner.n_rejected == 102


def test_dbscan_xy_angle_fan():
    coords = ['image_x', 'image_y', 'angle']
    X = fans1[coords].values
    dbscanner = clustering.DBScanner(X, min_samples=2)

    assert dbscanner.n_clusters == 6
    assert dbscanner.n_rejected == 15


def test_clustering_basics():
    cm.cluster_image_id(imid1, data=imid1data)
    assert cm.n_classifications == 94

    cm.cluster_image_id(imid2, data=imid2data)
    assert cm.n_classifications == 121

    for subdir in ['applied_cut_0.5', 'just_clustering']:
        expected = tdir / subdir
        assert expected.exists() and expected.is_dir()


def test_output_file_creation():
    for marking in ['blotches', 'fans', 'fnotches']:
        for ftype in ['.csv']:
            expected = tdir / (imid1 + '_' + marking + ftype)
            assert expected.exists()

    for marking in ['blotches']:
        for ftype in ['.csv']:
            expected = tdir / (imid2 + '_' + marking + ftype)
            if marking == 'blotches':
                assert expected.exists()
            else:  # 12q,i.e. imdid2 only has blotches
                assert not expected.exists()
