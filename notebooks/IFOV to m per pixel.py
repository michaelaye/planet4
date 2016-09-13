
# coding: utf-8

# $$ \tan\left(\frac{IFOV}{2}\right)=\frac{\frac{res}{2}}{altitude} $$

# Meaning...

# $$ res = 2\cdot{}altitude\cdot \tan\left(\frac{IFOV}{2}\right) $$

# In[ ]:

import math

def ifov_to_m_pix(ifov, alt=400e3):
    """expecting ifov in micro-radians"""
    return 2 * alt * math.tan(ifov*1e-6/2)
    


# In[ ]:

ifov_to_m_pix(5)


# In[ ]:



