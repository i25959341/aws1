from collections import deque

from PyGMO.problem import base
import rebound as reb
import numpy as np
import math
from simulator import SpaceSim
import os
import matplotlib.pyplot as plt

ROCKET = '-125544'
SUN_TO_EARTH = ['SUN',  'EARTH']
MARS_TO_PLUTO = ['MARS']

def get_simulation(sim_file, time_unit='day', integrator='ias15', include_rocket=False):
    '''
    If simulation file exists, load rebound simulation from it.
    Else create new rebound simulation with time unit.

    integrators = [ias15, whfast]
    '''

    sim_file = 'bin/' + sim_file + '.bin'

    if include_rocket:
        bodies = SUN_TO_EARTH + [ROCKET] + MARS_TO_PLUTO
    else:
        bodies = SUN_TO_EARTH + MARS_TO_PLUTO

    if not os.path.isfile(sim_file):

        sim = reb.Simulation()
        sim.units = time_unit, 'AU', 'Msun'

        sim.integrator = integrator
        if integrator == 'whfast':
            sim.gravity='basic'
            sim.dt = 1e-3

        for b in bodies:
            sim.add(b)

        sim.move_to_com()
        sim.save(sim_file)

        reb.OrbitPlot(sim)
        plt.show()
        del sim

    return reb.Simulation.from_file(sim_file)

sim = get_simulation("test", integrator='ias15', include_rocket=True)

MILLION_DAYS = 365

data = deque()
times = np.linspace(sim.t, sim.t + MILLION_DAYS, 10)

for idx, t in enumerate(times):
    sim.integrate(t)
    coord =  np.asarray([[b.x, b.y, b.z] for b in sim.particles])
    data.append(coord)

    print 'Integrated to time', sim.t

coords = np.array(data)

import sys
import time
from mpl_toolkits.mplot3d import Axes3D

BODIES = ['SUN', 'MERCURY', 'VENUS', 'EARTH', 'ROCKET', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO']

fig = plt.figure(figsize=(15,10))

ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.set_zlabel('z (AU)')

num_bodies = coords.shape[1]

ax.plot(coords[:,4,0],coords[:,4,1],coords[:,4,2], ':',linewidth=7, color='black', label='ROCKET')

# for i in [0,1,2,3,5,6,7,8,9,10]:
for i in [3]:
    ax.plot(coords[:,i, 0],coords[:,i,1],coords[:,i,2], '--',color='red', label=BODIES[i])

plt.show()
