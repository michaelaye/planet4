import math

import matplotlib.pyplot as plt
import numpy as np
from vpython import box, canvas, color, rate, sphere, vec

math.tau = np.tau = 2*math.pi


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


class FanSimulator(object):
    g_Forces = {'mars': np.array([0, 0, -3.80]),
                'earth': np.array([0, 0, -9.81])}
    radius = 0.1
    start = vec(0, 0, radius)
    win = 600
    L = 30.
    gray = vec(0.7, 0.7, 0.7)
    up = vec(0, 0, 1)

    @staticmethod
    def laminar_velocities(vmax, r, vent_radius):
        return vmax * (1 - r**2/vent_radius**2)

    @staticmethod
    def turbulent_velocities(vmax, r, vent_radius, n=7):
        return vmax * (1-r/vent_radius)**(1/n)

    def __init__(self, N=5000, vent_radius=0.5, vmax=50, dt=1e-2, location='mars',
                 is_turbulent=True, n=7):
        np.random.seed(42)
        self.N = N
        self.dt = dt
        self.vent_radius = vent_radius
        self.vmax = vmax
        self.particles = []
        self.t = None  # set to 0 in init_positions
        self.g = self.g_Forces[location]
        self.location = location
        self.is_turbulent = is_turbulent
        self.n = n

    def init_positions(self, vent_radius=None, N=None):
        if vent_radius is None:
            vent_radius = self.vent_radius
        if N is None:
            N = self.N
        radii = np.random.uniform(0, vent_radius, N)
        thetas = np.random.uniform(0, math.tau, N)
        X, Y = uniform_circle_sample(thetas, radii**2)
        self.positions = np.stack([X, Y, np.full_like(X, self.radius/2)], axis=1)
        self.radii = radii
        self.init_pos = self.positions.copy()
        self.t = 0

    def init_velocities(self, vmax=None, radii=None):
        if vmax is None:
            vmax = self.vmax
        if radii is None:
            radii = self.radii
        # using Hagen-Poiseulle flow's parabolic velocity distribution
        if not self.is_turbulent:
            vz = self.laminar_velocities(vmax, radii, self.vent_radius)
        else:
            vz = self.turbulent_velocities(vmax, radii, self.vent_radius, self.n)
        velocities = np.zeros((self.N, 3))
        # setting z-column to vz
        velocities[:, -1] = vz
        self.velocities = velocities
        self.init_v = velocities.copy()

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
                for p, pos in zip(self.particles, self.positions):
                    if p.update:
                        p.pos = vec(*pos)
                    if p.pos.z < self.start.z:
                        p.update = False

    def plot(self, ax=None, save=False, equal=True, xlim=None, ylim=None,
             **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        ax.scatter(self.positions[:, 0], self.positions[:, 1], 5,
                   **kwargs)
        if equal:
            ax.set_aspect('equal')
        ax.set_xlabel('Distance [m]')
        ax.set_ylabel('Spread [m]')
        ax.set_title("{0} particles, vent_radius={4} m, v0_z={1}, v0_x= {2}, jitter={3} [m/s]\n"
                     .format(self.N, self.vmax, self.incline, self.jitter, self.vent_radius))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if save:
            root = "/Users/klay6683/Dropbox/SSW_2015_cryo_venting/figures/"
            fig.savefig(root+'fan_vmax{}_incline{}_vent_radius{}.png'
                             .format(self.vmax, self.incline, self.vent_radius),
                        dpi=150)

    def init_vpython(self):
        scene = canvas(title="Fans", width=self.win, height=self.win, x=0, y=0,
                       center=vec(0, 0, 0), forward=vec(1, 0, -1),
                       up=self.up)
        scene.autoscale = False
        scene.range = 25

        h = 0.1
        _ = box(pos=vec(0, 0, -h/2), length=self.L, height=h, width=self.L,
                up=self.up, color=color.white)

        # create dust particles
        for pos in self.positions:
            p = sphere(pos=vec(*pos), radius=self.radius, color=color.red)
            p.update = True  # to determine if needs position update
            self.particles.append(p)
