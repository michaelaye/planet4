
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

radius = 0.1  # arbitrary for now
dt = 1e-2
start = vec(0, 0, radius)
g_M = np.array([0, 0, -3.80])
g_E = np.array([0, 0, -9.81])

N=3000

# positions
R = 0.5  # vent radius
radii = np.random.uniform(0, R, N)
thetas = np.random.uniform(0, math.tau, N)
X,Y = uniform_circle_sample(thetas, radii)
positions = np.stack([X, Y, np.full_like(X, radius/2)], axis=1)

# velocities
# Using parabolic profile for Hagen-Poiseulle flow
vmax = 50  # m/s
vz = vmax * (1 - radii**2/0.6**2)  # 2**2
velocities = np.zeros((N, 3))
velocities[:, -1] = vz
# incline the jet
velocities[:, 0] = 0.5  # incline along x-axis
# variations for vx and vy
radii = np.random.uniform(0, 0.2, N)
thetas = np.random.uniform(0, math.tau, N)
vx,vy = uniform_circle_sample(thetas, radii)
velocities[:, 0] += vx
velocities[:, 1] += vy
# define particles here to make loop work with or without
particles =[]
# save positions for later comparison
init_pos = positions.copy()


# In[ ]:

scene = canvas(title="Fans", width=win, height=win, x=0, y=0,
               center=vec(0, 0, 0), forward=vec(1,0,-1),
               up=up)
scene.autoscale = False
scene.range = 25

h = 0.1
mybox = box(pos=vec(0, 0, -h/2), length=L, height=h, width=L, up=up, color=color.white)

# create dust particles

for pos in positions:
    p = sphere(pos=vec(*pos), radius=radius, color=color.red)
    p.update = True  # to determine if needs position update
    particles.append(p)


# In[ ]:

t=0
while any(positions[:, -1] > 0):
    if particles:
        rate(200)
    
    # find all that are still above ground
    to_update = positions[:, -1] > 0
    positions[to_update] += velocities[to_update]*dt
    velocities[to_update] += g_M*dt
    if particles:
        for p,pos in zip(particles, positions):
            if p.update:
                p.pos = vec(*pos)
            if p.pos.z < start.z:
                p.update = False
    t+=dt


# In[ ]:

get_ipython().magic('matplotlib nbagg')

import seaborn as sns


# In[ ]:

fig, axes = plt.subplots(ncols=2)
axes[0].scatter(init_pos[:,0], init_pos[:,1], 5)
axes[1].scatter(positions[:,0], positions[:,1], 5)
for ax in axes:
    ax.set_aspect('equal')
    


# In[ ]:

t

