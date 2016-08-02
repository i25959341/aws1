from simulatorHigh import SpaceSim
import rebound
import numpy as np
import math
import matplotlib.pyplot as plt

def calculateDistance(sim2):
    rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
    marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
    distance = (rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2 +(rocketCord[2]-marsCord[2])**2

    return distance

x=SpaceSim()
deci = [[5.368545621384487e-06, 2.2642820923555172e-05, 0.00010579608330588702, 0.0], [250.00000536854563, 0.0013151829722556129, -0.002883903806489052, 0.0]]

dist = calculateDistance(x.simulate(deci))
print dist

steps = [
# 15,
# 10,
# 7.5,
5,
4.8,
3.84,
3.75,
2.5,
1.875,
1.25,
0.9375,
0.768,
0.625,
0.48,
0.46875,
0.3125,
0.234375,
0.15625,
0.1536,
0.1171875,
0.078125,
0.05859375,
0.048,
0.0390625,
# 0.03072,
# 0.029296875,
# 0.01953125,
# 0.006144,
# 0.0048
]
inx=[]
error=[]

for i, step in enumerate(steps):
    x=SpaceSim()
    deci = [[5.368545621384487e-06, 2.2642820923555172e-05, 0.00010579608330588702, 0.0], [250.00000536854563, 0.0013151829722556129, -0.002883903806489052, 0.0]]

    dummy = calculateDistance(x.simulateRefine(deci,step))
    inx.append(step)
    error.append(dummy-dist)

fig = plt.figure()
fig.suptitle('Error vs Timestep', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)

ax.set_xlabel('Timestep size')
ax.set_ylabel('Difference')
plt.xticks([x*2 for x in range(12)])
ax.plot(inx, error, "o")
#
plt.show()
