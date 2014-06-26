from planet4.markings import diffangle, rotate_vector
import numpy as np


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
