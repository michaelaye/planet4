
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

class FanSimulator(object):
    g_Forces={'mars':np.array([0, 0, -3.80]),
              'earth':np.array([0, 0, -9.81])}
    radius = 0.1
    start = vec(0, 0, radius)
    win = 600
    L = 30.
    gray = vec(0.7, 0.7, 0.7)
    up = vec(0, 0, 1)
    
    def __init__(self, N, vent_radius=0.5, vmax=50, dt=1e-2, location='mars'):
        np.random.seed(42)
        self.N = N
        self.dt = dt
        self.vent_radius = vent_radius
        self.vmax = vmax
        self.particles = []
        self.t = None  # set to 0 in init_positions
        self.g = self.g_Forces[location]
        
    def init_positions(self, vent_radius=None, N=None):
        if vent_radius is None:
            vent_radius = self.vent_radius
        if N is None:
            N = self.N
        radii = np.random.uniform(0, vent_radius, N)
        thetas = np.random.uniform(0, math.tau, N)
        X, Y = uniform_circle_sample(thetas, radii)
        self.positions = np.stack([X, Y, np.full_like(X, self.radius/2)], axis=1)
        self.radii = radii
        self.init_pos = self.positions.copy()
        self.t = 0
        
    def init_velocities(self, vmax=None):
        if vmax is None:
            vmax = self.vmax
        # using Hagen-Poiseulle flow's parabolic velocity distribution
        vz = vmax * (1 - self.radii**2/(self.vent_radius*1.05)**2)
        velocities = np.zeros((self.N, 3))
        # setting z-column to vz
        velocities[:, -1] = vz
        self.velocities = velocities
        
    def incline_and_vary_jet(self, incline=1, jitter=0.1):
        self.incline = incline
        self.velocities[:, 0] = incline
        self.jitter = jitter
        radii = np.random.uniform(0, jitter, self.N)
        thetas = np.random.uniform(0, math.tau, self.N)
        vx, vy = uniform_circle_sample(thetas, radii)
        self.velocities[:, 0] += vx
        self.velocities[:, 1] += vy
        
    def update(self):
        to_update = self.positions[:, -1] > 0
        self.positions[to_update] += self.velocities[to_update]*self.dt
        self.velocities[to_update] += self.g*self.dt
        self.t += self.dt
        
    @property
    def something_in_the_air(self):
        return any(self.positions[:, -1] > 0)
    
    def loop(self):
        while self.something_in_the_air:
            self.update()
            if self.particles:
                rate(200)
                for p,pos in zip(sim.particles, sim.positions):
                    if p.update:
                        p.pos = vec(*pos)
                    if p.pos.z < start.z:
                        p.update = False

    def plot(self, save=False, equal=True):
        fig, axes = plt.subplots(ncols=1, squeeze=False)
        axes = axes.ravel()
        axes[0].scatter(self.positions[:,0], self.positions[:,1], 5)
        for ax in axes:
            if equal:
                ax.set_aspect('equal')
            ax.set_xlabel('Distance [m]')
            ax.set_ylabel('Spread [m]')
        ax.set_title("{0} particles, v0_z={1}, v0_x= {2}, jitter={3} [m/s]\n"
                     "dt={4}"
                     .format(self.N, self.vmax, self.incline, self.jitter, self.dt))
        if save:
            root = "/Users/klay6683/Dropbox/SSW_2015_cryo_venting/figures/"
            fig.savefig(root+'fan_vmax{}_incline{}_vent_radius{}.png'
                             .format(self.vmax, self.incline, self.vent_radius),
                        dpi=150) 
    
    def init_vpython(self):
        scene = canvas(title="Fans", width=self.win, height=self.win, x=0, y=0,
               center=vec(0, 0, 0), forward=vec(1,0,-1),
               up=self.up)
        scene.autoscale = False
        scene.range = 25

        h = 0.1
        mybox = box(pos=vec(0, 0, -h/2), length=self.L, height=h, width=L, up=self.up,
                    color=color.white)

        # create dust particles
        for pos in self.positions:
            p = sphere(pos=vec(*pos), radius=self.radius, color=color.red)
            p.update = True  # to determine if needs position update
            self.particles.append(p)


# In[ ]:

#%matplotlib nbagg

import seaborn as sns
sns.set_context('notebook')


# In[ ]:

sim = FanSimulator(5000, vent_radius=0.1, dt=0.01)
sim.init_positions()
sim.init_velocities()
sim.incline_and_vary_jet(jitter=0.2, incline=10.0)

sim.loop()

sim.plot(save=True, equal=False)


# In[ ]:

sim = FanSimulator(5000, vent_radius=0.1, dt=0.001)
sim.init_positions()
sim.init_velocities()
sim.incline_and_vary_jet(jitter=0.2, incline=10.0)

sim.loop()

sim.plot(save=True, equal=False)


# In[ ]:

from pypet import Environment, cartesian_product


# In[ ]:

def add_parameters(traj, dt=1e-2):
    traj.f_add_parameter('N', 5000, comment='number of particles')
    traj.f_add_parameter('vent_radius', 0.5, comment='radius of particle emitting vent')
    traj.f_add_parameter('vmax', 50, comment='vmax in center of vent')
    traj.f_add_parameter('dt', dt, comment='dt of simulation')
    traj.f_add_parameter('incline', 10.0, comment='inclining vx value')
    traj.f_add_parameter('jitter', 0.1, comment='random x,y jitter for velocities')
    traj.f_add_parameter('location', 'mars', comment='location determining g-force')

def run_simulation(traj):
    sim = FanSimulator(traj.N, vent_radius=traj.vent_radius, vmax=traj.vmax,
                       dt=traj.dt, location=traj.location)
    sim.init_positions()
    sim.init_velocities()
    sim.incline_and_vary_jet(incline=traj.incline, jitter=traj.jitter)
    sim.loop()
    sim.plot(save=True, equal=False)
    traj.f_add_result('positions', sim.positions, comment='End positions of particles')
    traj.f_add_result('t', sim.t, comment='duration of flight')

env = Environment(trajectory='FanSimulation', filename='./pypet/',
                  large_overview_tables=True,
                  add_time=True,
                  multiproc=False,
                  ncores=6,
                  log_config='DEFAULT')

traj = env.v_trajectory

add_parameters(traj, dt=1e-2)

explore_dict = {'vent_radius':[0.1, 0.5, 1.0],
                'vmax':[10, 50, 100],
                'incline':[0.1, 1.0, 5.0]}

to_explore = cartesian_product(explore_dict)
traj.f_explore(to_explore)

env.f_run(run_simulation)

env.f_disable_logging()


# In[ ]:



