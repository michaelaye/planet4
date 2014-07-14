from planet4.markings import diffangle, rotate_vector, P4_Fan
import numpy as np
import pytest
import pandas as pd


def test_diffangle():
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    delta = 1e-3
    assert (diffangle(v1, v2, rads=False) - 90.0) < delta


def test_rotate_vector():
    v1 = np.array([1, 0])
    new_vec = rotate_vector(v1, 90)
    delta = 1e-3
    assert (diffangle(v1, new_vec, rads=False) - 90.0) < delta

    new_vec = rotate_vector(v1, 180)
    assert (diffangle(v1, new_vec, rads=False) - 180.0) < delta

    new_vec = rotate_vector(v1, 270)
    assert (diffangle(v1, new_vec, rads=False) - 270.0) < delta


class Test_P4_Fan:
    @pytest.fixture
    def datarow(self):
        # spread is half-spread here
        index = 'x y distance angle spread'.split()
        data = [0, 0, 1, 0, 45]
        return pd.Series(data=data, index=index)

    @pytest.fixture
    def fan(self, datarow):
        return P4_Fan(datarow)

    def test_base(self, fan):
        assert np.array_equal(fan.base, np.array([0, 0]))

    def test_length(self, fan):
        assert np.allclose(np.sqrt(0.5), fan.length)

    def test_v1(self, fan):
        assert np.allclose(np.array([0.5, -0.5]), fan.v1)

    def test_v2(self, fan):
        assert np.allclose(np.array([0.5, 0.5]), fan.v2)
