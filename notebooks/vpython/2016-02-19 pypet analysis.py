
# coding: utf-8

# In[ ]:

from pypet import Trajectory


# In[ ]:

traj = Trajectory(filename='./pypet/FanSimulation_2016_02_22_14h50m17s.hdf5')


# In[ ]:

traj.f_load(index=-1, load_parameters=2, load_results=2)


# In[ ]:

traj.f_get_parameters()


# In[ ]:

traj.f_get_explored_parameters()


# In[ ]:

def my_filter_function(location,dt):
    result = location =='mars' and dt=1e-2
    return result


# In[ ]:

set(traj.f_get('incline').f_get_range())


# In[ ]:

def filter_function(loc,dt,vmax,vent_radius,incline):
    result = loc=='mars' and dt==1e-2 and vmax==50        and vent_radius==0.5 and incline==1.0
    return result


# In[ ]:

def standard_filter(loc, dt, jitter, incline):
    result = loc=='mars' and dt==1e-2 and jitter==0.1 and incline!=10.0
    return result


# In[ ]:

idx_iter = traj.f_find_idx(['parameters.location','parameters.dt',
                            'parameters.jitter', 'parameters.incline'], standard_filter)


# In[ ]:

indexes = list(idx_iter)


# In[ ]:

get_ipython().magic('matplotlib nbagg')


# In[ ]:

def plot_data(i):

    traj.v_idx = i
    data = traj.res.crun.positions
    title = ''
    for k,v in traj.f_get_parameters(fast_access=True).items():
        key = k.split('.')[1]
        if key=='location' or key=='dt' or key=='N' or key=='jitter':
            continue
        t = "{}:{} | ".format(key, v)
        title += t
    fig, axes = plt.subplots(nrows=2, squeeze=False)
    axes = axes.ravel()
    for ax in axes:
        ax.scatter(data[:, 0], data[:, 1])
        ax.set_aspect('equal')
    axes[1].set_xlim(-15, 270)
    axes[1].set_ylim(-20, 20)
    fig.suptitle(title[:-2], fontsize=13)


# In[ ]:

get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set_context('notebook')


# In[ ]:

from ipywidgets import interact


# In[ ]:

interact(plot_data, i=(0, len(traj)-1));


# In[ ]:



