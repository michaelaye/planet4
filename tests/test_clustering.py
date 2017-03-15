from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pkg_resources as pr
import pytest

from planet4.dbscan import DBScanner


@pytest.fixture(scope='module')
def data():
    """Read data from test database csv file. """
    with pr.resource_stream('planet4', 'data/test_db.csv') as f:
        df = pd.read_csv(f)
    return df


@pytest.fixture(scope='module',
                params=['APF000012w', 'APF000012q'])
def img_id(request):
    imid = request.param
    return data[data.image_id == imid]


@pytest.fixture(scope='module')
def tdir():
    "Create a tempdir with teardown cleanup."
    _tdir = TemporaryDirectory()
    yield Path(_tdir.name)
    # teardown
    _tdir.cleanup()


def test_dbscanner_init():
    dbscanner = DBScanner()
    assert dbscanner.msf == 0.13
    assert isinstance(dbscanner.eps_values, dict)


def test_cluster_image_id():
    dbscan = DBScanner(save_results=False)
    dbscan.cluster_image_id('12w')
    assert dbscan.reduced_data['fan'].shape == (4, 15)
