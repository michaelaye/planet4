
# coding: utf-8

# # The sun is back!
# 
# I realised today that we are so close to southern spring time, I wondered if the sun already is up over Inca City. To do that I load first some of the tool libraries I have written.

# In[ ]:

from pymars import kmaspice


# Now I will create an object that knows how to deal with Martian times and illuminations.

# In[ ]:

inca = kmaspice.MarsSpicer()


# I saved some predefined places and their locations into the code, so that I don't need to remember the coordinates all the time. So let's justify the variable name by actually setting it on top of Inca City:

# In[ ]:

inca.goto('inca')


# By default, when I don't provide a time, the time is set to the current time. In the UTC timezone, that is:

# In[ ]:

inca.time.isoformat()


# To double-check how close we are to spring time in the southern hemisphere on Mars, I need to look at a value called L_s, which is the solar longitude.
# 
# This value measures the time of the seasons on Mars as its angular position during its trip around the sun which southern spring being at Ls = 180.

# In[ ]:

round(inca.l_s, 1)


# So, we are pretty close to spring then. But do we already have sunlight in Inca? We should remember that we are in polar areas, where we have darkness for half a year, just like on Earth. Let's have a look what is the local time in Inca:

# In[ ]:

inca.local_soltime


# Right, that's still in the night, so that most likely means that the sun is below the horizon, right?

# In[ ]:

round(inca.illum_angles.dsolar,1)


# Solar angles are measured from the local normal direction, with the sun directly over head being defined as 0. Which means the horizon is at 90 degrees. Hence, this value of 96 means the sun is below the horizon. But it is local night, so we would expect that!
# 
# Now comes the magic, let's just advance the time by a couple of hours, usually the sun is highest around local noon, so let's aim for that. I don't need to be precise now, so I go just 7 hours forward in time.
# Then I will have another look at the sun's angle.

# In[ ]:

inca.advance_time_by(7*3600)
round(inca.illum_angles.dsolar)


# Oh yes! This is just 2 degrees above the horizon, the sun is lurking over it just a tiny bit. But all you humans that work so much in helping us know what this means, right? Where there is sun, there is energy. And this energy can be used to sublime CO2 gas and create the wonderful fans we are studying.
# Let's make this a bit prettier with a bit more insight. First, let's go back the 7 hours again.

# In[ ]:

inca.advance_time_by(-7*3600)


# Now, I will create a loop with 100 elements, and check and write down the time each 10 minutes (= 600 seconds). I save the stuff in 2 new arrays to have it easier to plot things over time.

# In[ ]:

times = []
angles = []

for i in range(100):
    inca.advance_time_by(600)
    times.append(inca.local_soltime[3])
    angles.append(inca.illum_angles.dsolar)


# I'm now importing the pandas library, an amazing toolbox to deal with time-series data. Especially, the plots automatically get nicely formatted time-axes.

# In[ ]:

import pandas as pd


# In[ ]:

data = pd.Series(angles, index=times)


# I need to switch this notebook to show plots inside this notebook and not outside as an extra window, which is my default:
# 

# In[ ]:

get_ipython().magic('pylab inline')


# In[ ]:

data.plot()


# Here we see how the sun's angle is developing over time. As expected we see a minimum (i.e. highest sun over horizon) right around noon. 
# ###Do you hear the CO2 ice crackling?? ;) 
# I find it amazing to know that in a couple of hours some of our beloved fans are being created!
# 
# Next I wondered how long we already have the sun lurking over the horizon. For this I now will go backwards in 10 minute steps, but this time I take 2000 steps to cover more time. I then immediately plot the results:

# In[ ]:

times = []
angles = []

for i in range(2000):
    inca.advance_time_by(-600)
    times.append(inca.time)
    angles.append(inca.illum_angles.dsolar)
pd.Series(angles,index=times).plot()


# Interesting! Now we can see that around July 14th, the sun for the first time was below an inclination angle of 90, meaning that's the first day it lurked over the horizon. So for a week now, its energy is slowly building up CO2 gas underneath the CO2 ice cover that is everywhere. I could bet that at some place with weak thin ice, already the first fans are popping up. Unfortunately, the light conditions at 2-4 PM, when our favorite camera HiRISE is flying over these places, are still so bad, that we have to wait a few more days until imaging becomes possible. But be assured we are hot on the heels to catch the action as early as possible.
# 
# Stay tuned for more on this!
# 
