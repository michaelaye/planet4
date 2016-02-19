
# coding: utf-8

# In[ ]:

from vpython import *
from numpy import arange, array, newaxis, square, sum, sqrt
from math import pi, exp, sin, cos
import math


# In[ ]:

win=700

Nparticles = 10  # change this to have more or fewer particles

# Typical values
L = 1. # container is a cube L on a side
gray = vec(0.7,0.7,0.7)  # color of edges of container
Matom = 4e-3/6e23  # helium mass
Ratom = 0.03  # wildly exaggerated size of helium atom
dt = 1e-5


# In[ ]:

scene = canvas(title="Fans", width=win, height=win, x=0, y=0,
                center=vec(*3*[L/2]),
               forward=vec(1,0,0),
               up=vec(0,0,1))


# In[ ]:

xaxis = curve(pos=[vec(0,0,0), vec(L,0,0)], color=gray)
yaxis = curve(pos=[vec(0,0,0), vec(0,L,0)], color=gray)
zaxis = curve(pos=[vec(0,0,0), vec(0,0,L)], color=gray)
xaxis2 = curve(pos=[vec(L,L,L), vec(0,L,L), vec(0,0,L), vec(L,0,L)], color=gray)
yaxis2 = curve(pos=[vec(L,L,L), vec(L,0,L), vec(L,0,0), vec(L,L,0)], color=gray)
zaxis2 = curve(pos=[vec(L,L,L), vec(L,L,0), vec(0,L,0), vec(0,L,L)], color=gray)


# In[ ]:

particles = []
colors = [color.red, color.green, color.blue,
          color.yellow, color.cyan, color.magenta]
poslist = []
plist = []
mlist = []
rlist = []


# In[ ]:

from scipy.constants import k

T = 250
for i in range(Nparticles):
    Lmin = 1.1*Ratom
    Lmax = L-Lmin
    x = Lmin+(Lmax-Lmin)*random()
#     y = Lmin+(Lmax-Lmin)*random()
    y = L/2
    z = 0
    r = Ratom
    particles = particles+[sphere(pos=vec(x,y,z), radius=r, color=colors[i % 6])]
    mass = Matom*r**3/Ratom**3
    pavg = sqrt(2.*mass*1.5*k*T) # average kinetic energy p**2/(2mass) = (3/2)kT
#     theta = np.deg2rad(10)
#     phi = 2*pi*random()
#     px = pavg*sin(theta)*cos(phi)
#     py = pavg*sin(theta)*sin(phi)
#     pz = pavg*cos(theta)
    px = 0
    py = 0
    pz = pavg
    poslist.append((x,y,z))
    plist.append((px,py,pz))
    mlist.append(mass)
    rlist.append(r)


# In[ ]:

pos = array(poslist)
p = array(plist)
m = array(mlist)
m.shape = (Nparticles,1)
radius = array(rlist)

pos = pos+(p/m)*(dt/2.) # initial half-step


# In[ ]:

while True:
    rate(100)

    # Update all positions
    pos = pos+(p/m)*dt

    r = pos-pos[:,newaxis] # all pairs of atom-to-atom vectors
    rmag = sqrt(sum(square(r),-1)) # atom-to-atom scalar distances
    hit = np.less_equal(rmag,radius+radius[:,None])-np.identity(Nparticles)
    hitlist = np.sort(np.nonzero(hit.flat)[0]).tolist() # i,j encoded as i*Nparticles+j

    # If any collisions took place:
    for ij in hitlist:
        i, j = divmod(ij,Nparticles) # decode atom pair
        hitlist.remove(j*Nparticles+i) # remove symmetric j,i pair from list
        ptot = p[i]+p[j]
        mi = m[i,0]
        mj = m[j,0]
        vi = p[i]/mi
        vj = p[j]/mj
        ri = particles[i].radius
        rj = particles[j].radius
#         a = mag(vj-vi)**2
        a = (vj-vi).dot(vj-vi)
        if a < 1e-3:
            continue # exactly same velocities
        b = 2*dot(pos[i]-pos[j],vj-vi)
        c = (pos[i]-pos[j]).dot(pos[i]-pos[j]) - (ri+rj)**2
        d = b**2-4.*a*c
        if d < 0:
            continue # something wrong; ignore this rare case
        deltat = (-b+math.sqrt(d))/(2.*a) # t-deltat is when they made contact
        pos[i] = pos[i]-(p[i]/mi)*deltat # back up to contact configuration
        pos[j] = pos[j]-(p[j]/mj)*deltat
        mtot = mi+mj
        pcmi = p[i]-ptot*mi/mtot # transform momenta to cm frame
        pcmj = p[j]-ptot*mj/mtot
        rrel = math.sqrt((pos[j]-pos[i]).dot(pos[j]-pos[i]))
        pcmi = pcmi-2*dot(pcmi,rrel)*rrel # bounce in cm frame
        pcmj = pcmj-2*dot(pcmj,rrel)*rrel
        p[i] = pcmi+ptot*mi/mtot # transform momenta back to lab frame
        p[j] = pcmj+ptot*mj/mtot
        pos[i] = pos[i]+(p[i]/mi)*deltat # move forward deltat in time
        pos[j] = pos[j]+(p[j]/mj)*deltat
 
    # Bounce off walls
    outside = np.less_equal(pos,Ratom) # walls closest to origin
    p1 = p*outside
    p = p-p1+abs(p1) # force p component inward
    outside = np.greater_equal(pos,L-Ratom) # walls farther from origin
    p1 = p*outside
    p = p-p1-abs(p1) # force p component inward

    # Update positions of canvas objects
    for i in range(Nparticles):
        particles[i].pos = vec(*pos[i])


# In[ ]:



