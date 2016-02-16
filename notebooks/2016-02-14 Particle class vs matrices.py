
# coding: utf-8

# In[ ]:

class Particle(object):
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    def __repr__(self):
        return "x={}\ny={}\nz={}".format(self.x, self.y, self.z)
    def apply_lateral_wind(self, dx, dy):
        self.x += dx
        self.y += dy

start_values = np.random.random((int(1e6),3))

particles = [Particle(*i) for i in start_values]


# In[ ]:

get_ipython().magic('timeit _ = [p.apply_lateral_wind(0.5, 1.2) for p in particles]')


# In[ ]:

get_ipython().magic('timeit start_values[...,:2] += np.array([0.5,1.2])')


# In[ ]:

class NumpyParticle(object):
    def __init__(self, coords):
        self.coords = coords
    @property
    def x(self):
        return self.coords[0]
    @property
    def y(self):
        return self.coords[1]
    @property
    def z(self):
        return self.coords[2]

    def __repr__(self):
        return "x={}\ny={}\nz={}".format(self.x, self.y, self.z)
    def apply_lateral_wind(self, dx, dy):
        self.coords[0] += dx
        self.coords[1] += dy

start_values = np.random.random((int(1e6), 3))
numpyparticles = [NumpyParticle(i) for i in start_values]


# In[ ]:

get_ipython().magic('timeit _ = [p.apply_lateral_wind(0.5, 1.2) for p in numpyparticles]')


# In[ ]:

class Particles(object):
    def __init__(self, coords):
        self.coords = coords

    def __repr__(self):
        return "Particles(coords={})".format(self.coords)

    def apply_lateral_wind(self, dx, dy):
        self.coords[:, 0] += dx
        self.coords[:, 1] += dy

    def apply_lateral_wind2(self, dx, dy):
        self.coords[..., :2] += [dx, dy]

start_values = np.random.random((int(1e6), 3))        
particles = Particles(start_values)


# In[ ]:

get_ipython().magic('timeit particles.apply_lateral_wind(0.5, 1.2)')


# In[ ]:

get_ipython().magic('timeit particles.apply_lateral_wind2(0.5, 1.2)')


# In[ ]:

import numpy as np
from numba import jitclass          # import the decorator
from numba import int32, float32, float64    # import the types

spec = [
    ('x', float32),               # a simple scalar field
    ('y', float32),               # a simple scalar field
    ('z', float32),               # a simple scalar field
]

@jitclass(spec)
class NumbaParticle(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def apply_lateral_wind(self, dx, dy):
        self.x += dx
        self.y += dy


# In[ ]:

particles = [NumbaParticle(*i) for i in start_values]


# In[ ]:

start_values = np.random.random((int(1e6), 3))


# In[ ]:

spec = [
    ('coords', float64[:,:]),
]

@jitclass(spec)
class NumbaParticles(object):
    def __init__(self, coords):
        self.coords = coords

    def apply_lateral_wind(self, dx, dy):
        self.coords[:, 0] += dx
        self.coords[:, 1] += dy

    def apply_lateral_wind2(self, dx, dy):
        self.coords[:, :2] += np.array([dx, dy], dtype=float64)


# In[ ]:

particles = NumbaParticles(start_values)


# In[ ]:

get_ipython().magic('timeit particles.apply_lateral_wind(0.5, 1.2)')


# In[ ]:

get_ipython().magic('timeit particles.apply_lateral_wind2(0.5, 1.2)')


# In[ ]:

from planet4 import io


# In[ ]:

db = io.DBManager()


# In[ ]:

data = db.get_image_name_markings("PSP_003092_0985")


# In[ ]:

blotches = data[data.marking=='blotch']


# In[ ]:

blotches[(blotches.radius_1<10) | (blotches.radius_2<10)]


# In[ ]:

get_ipython().magic('matplotlib inline')


# In[ ]:

blotches[blotches.radius_2 <20].radius_1.hist(bins=100)


# In[ ]:

fans = data[data.marking=='fan']


# In[ ]:

fans.columns


# In[ ]:

plt.figure()
fans[fans.distance<50].distance.value_counts()


# In[ ]:

fans.angle.hist(bins=100)


# In[ ]:

fans.columns


# In[ ]:

fans[fans.spread<50].spread.hist(bins=100)


# In[ ]:

fans[fans.spread<50].spread.value_counts().head()


# In[ ]:

fans[fans.spread<10].distance.hist(bins=50)


# In[ ]:



