
# coding: utf-8

# In[ ]:

from vpython import sphere, canvas, box, vec, color, rate
import math
math.tau = np.tau = 2*math.pi


# In[ ]:

def cart2pol(vec):
    theta = np.arctan2(vec[:, 1], vec[:, 0])
    rho = np.hypot(vec[:, 0], vec[:, 1])
    return theta, rho

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def uniform_circle_sample(theta, rho):
    x = np.sqrt(rho) * np.cos(theta)
    y = np.sqrt(rho) * np.sin(theta)
    return x, y


# In[ ]:

win=600
L = 30. # container is a cube L on a side
gray = vec(0.7,0.7,0.7)  # color of edges of container
up = vec(0, 0, 1)


# In[ ]:

scene = canvas(title="Fans", width=win, height=win, x=0, y=0,
               center=vec(0, 0, 0), forward=vec(1,0,-1),
               up=up)
scene.autoscale = False
scene.range = 25

h = 0.1
mybox = box(pos=vec(0, 0, -h/2), length=L, height=h, width=L, up=up, color=color.white)

m = 1  # kg
radius = 0.1  # arbitrary for now
dt = 1e-2
start = vec(0, 0, radius)
g_M = np.array([0, 0, -3.80])
g_E = np.array([0, 0, -9.81])
Fg = m*g_E

N=1000
# positions
radii = np.random.uniform(-2, 2, N)
thetas = np.random.uniform(0, np.tau, N)
X,Y = uniform_circle_sample(thetas, radii)
positions = np.stack([X, Y, np.full_like(X, radius/2)], axis=1)
# velocities
radii = np.random.uniform(-1, 1, N)
thetas = np.random.uniform(0, np.tau, N)
VX, VY = pol2cart(thetas, radii)
velocities = np.stack([VX, VY, np.full_like(X, 20)], axis=1)
# create dust particles
particles =[]
for pos in positions:
    p = sphere(pos=vec(*pos), radius=radius, color=color.red)
    p.update = True  # to determine if needs position update
    particles.append(p)


# In[ ]:

t=0
while True:
    rate(200)
    
    # update position first
    positions += velocities*dt
    velocities += g_E*dt
    for p,pos in zip(particles, positions):
        if p.update:
            p.pos = vec(*pos)
        if p.pos.z < start.z:
            p.update = False
    t+=dt
    if all([not p.update for p in particles]):
        print('Done.')
        break


# In[ ]:

get_ipython().magic('matplotlib nbagg')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

mean = [0,0]
cov = [[1, 0], [0, 1]]


# In[ ]:

x, y = np.random.multivariate_normal(mean, cov, 5000).T


# In[ ]:

plt.figure(figsize=(6,6))
plt.plot(x, y, 'x')
# plt.axis('equal')


# In[ ]:



