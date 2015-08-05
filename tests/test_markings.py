from planet4.markings import diffangle, rotate_vector, Fan
import numpy as np
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


class Test_Fan:
    index = 'x y distance angle spread'.split()
     # spread is full spread here

    def test_along_x(self):
        data = [0, 0, 1, 0, 90]
        datarow = pd.Series(data=data, index=self.index)
        fan = Fan(datarow)

        assert np.array_equal(fan.base, np.array([0, 0]))

        assert np.allclose(np.sqrt(0.5), fan.length)

        assert np.allclose(np.array([0.5, -0.5]), fan.v1)

        assert np.allclose(np.array([0.5, 0.5]), fan.v2)

    def test_along_y(self):
        data = [0, 0, 2, 90, 90]
        fan = Fan(pd.Series(data=data, index=self.index))

        assert np.array_equal(fan.base, np.array([0, 0]))

        assert np.allclose(np.sqrt(2), fan.length)

        assert np.allclose(np.array([1.0, 1.0]), fan.v1)

        assert np.allclose(np.array([-1.0, 1.0]), fan.v2)

    def test_along_first_sector(self):
        data = [0, 0, np.sqrt(2), 45, 90]
        fan = Fan(pd.Series(data=data, index=self.index))

        assert np.array_equal(fan.base, np.array([0, 0]))

        assert np.allclose(1.0, fan.length)

        assert np.allclose(np.array([1.0, 0.0]), fan.v1)

        assert np.allclose(np.array([0.0, 1.0]), fan.v2)
