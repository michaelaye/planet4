
# coding: utf-8

# In[ ]:

from vpython import sphere, canvas, box, vec, color, rate


# In[ ]:

win=600

L = 10. # container is a cube L on a side

gray = vec(0.7,0.7,0.7)  # color of edges of container

up = vec(0, 0, 1)


# In[ ]:

scene = canvas(title="Fans", width=win, height=win, x=0, y=0,
               center=vec(0, 0, 0), forward=vec(1,0,-1),
               up=up)
scene.autoscale = False
scene.range = 10

h = 0.1
mybox = box(pos=vec(0, 0, -h/2), length=L, height=h, width=L, up=up, color=gray)

m = 1  # kg
radius = 0.5  # arbitrary for now
dt = 1e-2
start = vec(0, 0, radius)
v0 = vec(0, 1, 10)
g = vec(0, 0, -9.81)
Fg = m*g
particles =[]

for position, c in zip([start, start+vec(0, 3*radius, 0)],
                           [color.red, color.blue]):
    p = sphere(pos=position, radius=radius, color=c)
    p.v = v0
    p.update = True  # to determine if needs position update
    particles.append(p)


# In[ ]:

while True:
    rate(100)
    
    # update position first
    for p in particles:
        if p.update:
            p.pos += p.v*dt
        if p.pos.z < start.z:
            p.update = False
        p.v += g*dt
    
    if all([not p.update for p in particles]):
        break


# In[ ]:



