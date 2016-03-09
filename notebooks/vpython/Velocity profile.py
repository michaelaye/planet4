
# coding: utf-8

# In[ ]:

vent_radius = 0.5
r = np.linspace(-vent_radius, vent_radius, 100)


# In[ ]:

get_ipython().magic('matplotlib nbagg')


# In[ ]:

def v_prof(vmax, r, factor):
    return vmax * (1 - factor*(r/(vent_radius))**2)

def v_prof_turb(vmax, r, factor):
    return v_prof(vmax, r, factor)

def v_power_law(vmax, r, n):
    
    return vmax* (1-r/vent_radius)**(1/n)


# In[ ]:

def get_v(r, power):
    def v_power_law(vmax, r, n):
        return vmax* (1-r/vent_radius)**(1/n)
    v = np.zeros_like(r)
    v[r>0] = v_power_law(vmax, r[r>0], power)
    v[r<0] = v_power_law(vmax, np.abs(r[r<0]), power)
    return v


# In[ ]:

vmax = 100
plt.plot(r, v_prof(vmax, r, 1), label='1')
for i in range(8,10, 1):
    plt.plot(r, get_v(r, i), label=i)
plt.legend()


# In[ ]:



