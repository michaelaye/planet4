from planet4.markings import diffangle, rotate_vector, Fan
import numpy as np
import pandas as pd


# from planet4 import markings
# import numpy as np
# from numpy.testing import assert_allclose
# import pandas as pd
#
# blotchdata = dict(
#     x = 100,
#     y = 200,
#     radius_1 = 30,
#     radius_2 = 40,
#     angle = 25,
#     image_id='blotch_id',
# )
# blotchdata = pd.Series(blotchdata)
#
# fandata = dict(x = 300,
#     y = 400,
#     angle = 55,
#     spread = 17,
#     distance = 23,
#     image_id = 'fan_id'
# )
# fandata = pd.Series(fandata)
#
#
# # test_blotch
# blotch = markings.Blotch(blotchdata)
# assert blotch.angle == 25
# assert blotch.center == (100, 200)
# assert blotch.height == 80
# assert blotch.width == 60
# assert blotch.x == 100
# assert blotch.y == 200
#
#
# # test_fan
# fan = markings.Fan(fandata)
# assert_allclose(fan.base, np.array([300, 400]))
# assert_allclose(fan.coords, np.array([[ 313.92663903,  414.67561542],
#                                         [ 300.        ,  400.        ],
#                                         [ 309.02737644,  418.10611602]]))
# assert_allclose(fan.inside_half, 8.5)
# assert_allclose(fan.armlength, 20.231781009871817)
# assert_allclose(fan.v1, np.array([13.92663903, 14.67561542]))
# assert_allclose(fan.v2, np.array([  9.02737644,  18.10611602]))
# assert_allclose(fan.center, np.array([ 311.47700774,  416.39086572]))
# assert_allclose(fan.circle_base, np.array([ 4.89926259, -3.4305006 ]))
# assert_allclose(fan.radius, 2.990447637172394)
# assert_allclose(fan.center, np.array([ 311.47700774,  416.39086572]))
# assert_allclose(fan.midpoint, np.array([ 306.65986158,  409.51126803]))
# assert_allclose(fan.base_to_midpoint_vec,
#                 np.array([[ 300.        ,  400.        ],
#                           [ 306.65986158,  409.51126803]]))
#
#
# fnotch = markings.Fnotch(0.4, markings.Fan(fandata),
#                          markings.Blotch(blotchdata))
# assert_allclose(fnotch.value, 0.4)
# assert isinstance(fnotch.get_marking(0.8), markings.Blotch)
# assert isinstance(fnotch.get_marking(0.3), markings.Fan)


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

        assert np.allclose(np.sqrt(0.5), fan.armlength)

        assert np.allclose(np.array([0.5, -0.5]), fan.v1)

        assert np.allclose(np.array([0.5, 0.5]), fan.v2)

    def test_along_y(self):
        data = [0, 0, 2, 90, 90]
        fan = Fan(pd.Series(data=data, index=self.index))

        assert np.array_equal(fan.base, np.array([0, 0]))

        assert np.allclose(np.sqrt(2), fan.armlength)

        assert np.allclose(np.array([1.0, 1.0]), fan.v1)

        assert np.allclose(np.array([-1.0, 1.0]), fan.v2)

    def test_along_first_sector(self):
        data = [0, 0, np.sqrt(2), 45, 90]
        fan = Fan(pd.Series(data=data, index=self.index))

        assert np.array_equal(fan.base, np.array([0, 0]))

        assert np.allclose(1.0, fan.armlength)

        assert np.allclose(np.array([1.0, 0.0]), fan.v1)

        assert np.allclose(np.array([0.0, 1.0]), fan.v2)
