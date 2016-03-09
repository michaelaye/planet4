
# coding: utf-8

# In[ ]:

from planet4 import FanSimulator


# In[ ]:

from planet4.fansimulator import cart2pol


# In[ ]:

get_ipython().magic('matplotlib inline')

import seaborn as sns
sns.set_context('notebook', font_scale=1.3)


# In[ ]:

r = np.linspace(0, 0.5, 100)


# In[ ]:

v_laminar = FanSimulator.laminar_velocities(50, r, 0.5)


# In[ ]:

v_turb = FanSimulator.turbulent_velocities(50, r, 0.5)


# In[ ]:

plt.plot(r, v_laminar, label='laminar')
plt.plot(r, v_turb, label='turb')
plt.legend()


# In[ ]:

simturb = FanSimulator(N=50000, vent_radius=0.5, vmax=50, is_turbulent=True)
simturb.init_positions()
simturb.init_velocities()
simturb.incline_and_vary_jet(jitter=0.05, incline=2)
simturb.loop()


# In[ ]:

simlam = FanSimulator(N=50000, vent_radius=0.5, vmax=50, is_turbulent=False)
simlam.init_positions()
simlam.init_velocities()
simlam.incline_and_vary_jet(jitter=0.05, incline=2)
simlam.loop()


# In[ ]:

fig, axes = plt.subplots(nrows=2, sharex=True)
simlam.plot(ax=axes[0], alpha=0.02, label='laminar' )
axes[0].set_xlabel('')
simturb.plot(ax=axes[1], alpha=0.02, label='turbulent')
for ax in axes:
    ax.legend(fontsize=12)


# In[ ]:

fig = plt.figure(figsize=(14,4))
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))
ax1.plot(r, v_laminar, '-', color='blue', label='laminar')
ax1.plot(r, v_turb, '--', color='blue', label='turbulent')
ax1.set_xlabel('Vent radius [m]')
ax1.set_ylabel(r'$v_z$', fontsize=20)
simlam.plot(ax=ax2, xlim=(-10, 80), ylim=(-8,8), alpha=0.02, label='laminar',color='blue')
axes[0].set_xlabel('')
simturb.plot(ax=ax3, xlim=(-10, 80), ylim=(-8,8), alpha=0.02, label='turbulent', color='blue')
for ax in [ax1, ax2, ax3]:
    ax.legend(fontsize=12)
ax2.set_xlabel('')
t = ax3.get_title()
fig.suptitle(t)
yloc = plt.MaxNLocator(4)
for ax in [ax2, ax3]:
    ax.set_title('')
    ax.yaxis.set_major_locator(yloc)
fig.tight_layout()
fig.subplots_adjust(top=0.9)
fig.savefig('/Users/klay6683/Dropbox/SSW_2015_cryo_venting/figures/preliminary_jets.png', dpi=200)


# In[ ]:

sim_no_jitter = FanSimulator(N=10000, vent_radius=0.5, vmax=10, is_turbulent=True)
sim_no_jitter.init_positions()
sim_no_jitter.init_velocities()
sim_no_jitter.incline_and_vary_jet(jitter=0.00, incline=2)
sim_no_jitter.loop()


# In[ ]:

sim_with_jitter = FanSimulator(N=10000, vent_radius=0.5, vmax=10, is_turbulent=True)
sim_with_jitter.init_positions()
sim_with_jitter.init_velocities()
sim_with_jitter.incline_and_vary_jet(jitter=0.05, incline=2)
sim_with_jitter.loop()


# In[ ]:

fig, axes = plt.subplots(nrows=2, figsize=(12,6), sharex=True)
xlim = (-2, 13)
ylim = (-1.5, 1.5)
sim_no_jitter.plot(ax=axes[0], xlim=xlim, ylim=ylim, alpha=0.3)
sim_with_jitter.plot(ax=axes[1], xlim=xlim, ylim=ylim, alpha=0.3)
axes[0].set_xlabel('')
fig.tight_layout()
fig.savefig("/Users/klay6683/Dropbox/SSW_2015_cryo_venting/figures/jitter_example.png",
            dpi=200)


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



