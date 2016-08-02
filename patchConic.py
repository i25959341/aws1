import rebound
import numpy as np
import math

# In[2]:

sim = rebound.Simulation()
sim.units = ('hr', 'AU', 'Msun')
sim.integrator = "ias15"

sim.add(m=1e-6)
# sim.add(m=1e-12,x=-0.289672967317, y=-0.968608131551, z=-0.00012550348302, vx=0.941039046412, vy=0.258782971916, vz=6.91937898955e-06)
sim.add(m=1e-34,a=0.00001)

print sim.status()
sim.move_to_com()
print sim.status()


# In[3]:

from IPython.display import display, clear_output
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib')
fig = rebound.OrbitPlot(sim)


# In[4]:

ps = sim.particles
tau = 1000.
def migrationForce(reb_sim):
    ps[1].ax += 0.0001 # AU/HOUR^2
    ps[1].ay += 0.0001 # AU/HOUR^2
    ps[1].az += 0.0
    deltaV = (ps[1].ax**2.0+ps[1].ay**2.0+ps[1].az**2.0)**0.5
    ps[1].m = ps[1].m*math.exp(-deltaV/100000000.)  # Mass loss

def stop(reb_sim):
    ps[1].ax = 0.0
    ps[1].ay = 0.0
    ps[1].az = 0.0
#     ps[1].m=ps[1].m*0.9


# In[5]:

sim.additional_forces = migrationForce
sim.force_is_velocity_dependent = 1
print sim.status()

# In[6]:

Nout = 100
a_s = np.zeros(Nout)
times = np.linspace(0.,10.*2.*np.pi,Nout)
for i, time in enumerate(times):

    if i>5:
        sim.additional_forces = stop

    rocketCord = (sim.particles[0].x,sim.particles[0].y,sim.particles[0].z)
    earthCord = (sim.particles[1].x,sim.particles[1].y,sim.particles[1].z)
    distance = (rocketCord[0]-earthCord[0])**2 +(rocketCord[1]-earthCord[1])**2 +(rocketCord[2]-earthCord[2])**2

    if distance>0.01:
        break
    sim.integrate(time)
    a_s[i] = sim.particles[1].a

    # fig = rebound.OrbitPlot(sim)
    # display(fig)
    # plt.close(fig)
    # clear_output(wait=True)


# In[7]:

print sim.status()

# In[8]:

sim2 = rebound.Simulation()
sim2.units = ('hr', 'AU', 'Msun')
sim2.integrator = "ias15"


# In[9]:

sim2.add(m=1.)
sim2.add(m=1e-6,a=1.)
sim2.add(m=1e-6,a=2., Omega = 2.35)
sim2.move_to_com()


# In[10]:

Nout = 10
a_s = np.zeros(Nout)
times = np.linspace(0.,sim.t,Nout)
for i, time in enumerate(times):
    sim2.integrate(time)
    # fig = rebound.OrbitPlot(sim2)
    # display(fig)
    # plt.close(fig)
    # clear_output(wait=True)


# In[11]:

sim2.status()


# In[12]:

rocket = sim.particles[1]
earth = sim2.particles[1]

rocketx = rocket.x+earth.x
rockety = rocket.y+earth.y
rocketz = rocket.z+earth.z

rocketvx = rocket.vx+earth.vx
rocketvy = rocket.vy+earth.vy
rocketvz = rocket.vz+earth.vz


# In[13]:

sim2.add(m=1e-34, x = rocketx, y =rockety,z = rocketz,vx=rocketvx,vy=rocketvy,vz= rocketvz)
sim2.move_to_com()



# In[14]:

ps = sim2.particles

def migrationForce2(reb_sim):
    ps[3].ax += 0.0001 #Thrust in X
    ps[3].ay += 0.0001 #Thrust in Y
    ps[3].az += 0.0
    deltaV = (ps[3].ax**2.0+ps[3].ay**2.0+ps[3].az**2.0)**0.5
#     ps[3].m = ps[3].m*math.exp(-deltaV)  # Mass loss

def migrationForceNorth(reb_sim):
    ps[3].ax += 0.0001 #Thrust in X
    ps[3].ay += 0.0001 #Thrust in Y
    ps[3].az += 0.0
    deltaV = (ps[3].ax**2.0+ps[3].ay**2.0+ps[3].az**2.0)**0.5
#     ps[3].m = ps[3].m*math.exp(-deltaV)  # Mass loss

def migrationForceNorthEast(reb_sim):
    ps[3].ax += 0.0001 #Thrust in X
    ps[3].ay += 0.0001 #Thrust in Y
    ps[3].az += 0.0
    deltaV = (ps[3].ax**2.0+ps[3].ay**2.0+ps[3].az**2.0)**0.5
#     ps[3].m = ps[3].m*math.exp(-deltaV)  # Mass loss

def migrationForce2East(reb_sim):
    ps[3].ax += 0.0001 #Thrust in X
    ps[3].ay += 0.0001 #Thrust in Y
    ps[3].az += 0.0
    deltaV = (ps[3].ax**2.0+ps[3].ay**2.0+ps[3].az**2.0)**0.5
#     ps[3].m = ps[3].m*math.exp(-deltaV)  # Mass loss

def migrationForce2EastSouth(reb_sim):
    ps[3].ax += 0.0001 #Thrust in X
    ps[3].ay += 0.0001 #Thrust in Y
    ps[3].az += 0.0
    deltaV = (ps[3].ax**2.0+ps[3].ay**2.0+ps[3].az**2.0)**0.5
#     ps[3].m = ps[3].m*math.exp(-deltaV)  # Mass loss

def migrationForceSouth(reb_sim):
    ps[3].ax += 0.0001 #Thrust in X
    ps[3].ay += 0.0001 #Thrust in Y
    ps[3].az += 0.0
    deltaV = (ps[3].ax**2.0+ps[3].ay**2.0+ps[3].az**2.0)**0.5
#     ps[3].m = ps[3].m*math.exp(-deltaV)  # Mass loss

def migrationForce2SouthWest(reb_sim):
    ps[3].ax += 0.0001 #Thrust in X
    ps[3].ay += 0.0001 #Thrust in Y
    ps[3].az += 0.0
    deltaV = (ps[3].ax**2.0+ps[3].ay**2.0+ps[3].az**2.0)**0.5
#     ps[3].m = ps[3].m*math.exp(-deltaV)  # Mass loss

def migrationForce2West(reb_sim):
    ps[3].ax += 0.0001 #Thrust in X
    ps[3].ay += 0.0001 #Thrust in Y
    ps[3].az += 0.0
    deltaV = (ps[3].ax**2.0+ps[3].ay**2.0+ps[3].az**2.0)**0.5
#     ps[3].m = ps[3].m*math.exp(-deltaV)  # Mass loss


def stag(reb_sim):
    c=0.9
    vEjection=0.0000001
    deltaV = vEjection*math.log(1/c)
    ps[3].ax += deltaV #Thrust in X
    ps[3].m=ps[3].m*c # Model staging

def stop2(reb_sim):
    ps[3].ax = 0.0
    ps[3].ay = 0.0
    ps[3].az = 0.0
#     ps[1].m=ps[1].m*0.9


# In[15]:

# sim2.additional_forces = migrationForce2
# sim2.force_is_velocity_dependent = 1


# In[16]:

Nout = 500
a_s = np.zeros(Nout)
times = np.linspace(0.,5000.*2.*np.pi,Nout)

for i, time in enumerate(times):

    rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
    marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
    distance = (rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2 +(rocketCord[2]-marsCord[2])**2

    if distance<0.01:
        break

    sim2.integrate(time)
    # fig = rebound.OrbitPlot(sim2)
    # display(fig)
    # plt.close(fig)
    # clear_output(wait=True)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:
