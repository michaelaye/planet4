
# coding: utf-8

# In[ ]:

# setup
from planet4 import markings
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

blotchdata = dict(
    x = 100,
    y = 200,
    radius_1 = 30,
    radius_2 = 40,
    angle = 25,
    image_id='blotch_id',
)
blotchdata = pd.Series(blotchdata)

fandata = dict(x = 300,
    y = 400,
    angle = 55,
    spread = 17,
    distance = 23,
    image_id = 'fan_id'
)
fandata = pd.Series(fandata)


# In[ ]:

# test_blotch
blotch = markings.Blotch(blotchdata)
assert blotch.angle == 25
assert blotch.center == (100, 200)
assert blotch.height == 80
assert blotch.width == 60
assert blotch.x == 100
assert blotch.y == 200


# In[ ]:

# test_fan
fan = markings.Fan(fandata)
assert_allclose(fan.base, np.array([300, 400]))
assert_allclose(fan.coords, np.array([[ 313.92663903,  414.67561542],
                                        [ 300.        ,  400.        ],
                                        [ 309.02737644,  418.10611602]]))
assert_allclose(fan.inside_half, 8.5)
assert_allclose(fan.armlength, 20.231781009871817)
assert_allclose(fan.v1, np.array([13.92663903, 14.67561542]))
assert_allclose(fan.v2, np.array([  9.02737644,  18.10611602]))
assert_allclose(fan.center, np.array([ 311.47700774,  416.39086572]))
assert_allclose(fan.circle_base, np.array([ 4.89926259, -3.4305006 ]))
assert_allclose(fan.radius, 2.990447637172394)
assert_allclose(fan.center, np.array([ 311.47700774,  416.39086572]))
assert_allclose(fan.midpoint, np.array([ 306.65986158,  409.51126803]))
assert_allclose(fan.base_to_midpoint_vec,
                np.array([[ 300.        ,  400.        ],
                          [ 306.65986158,  409.51126803]]))


# In[ ]:

fnotch = markings.Fnotch(0.4, markings.Fan(fandata),
                         markings.Blotch(blotchdata))
assert_allclose(fnotch.value, 0.4)
assert isinstance(fnotch.get_marking(0.8), markings.Blotch)
assert isinstance(fnotch.get_marking(0.3), markings.Fan)


# In[ ]:



