from simulatorHigh import SpaceSim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

deci = [[171.02740866424318, -0.00027481006013446203, -0.0024876012556708225, -5.940960457227606e-05]]

z=[]
x=[]
y=[]

time =171.02740866424318

z1=[]
x1=[]
y1=[]

Patch=0.0018

deltaX=np.linspace(-0.00027481006013446203-Patch,-0.00027481006013446203+Patch,50)
deltaY=np.linspace(-0.0024876012556708225-Patch,-0.0024876012556708225+Patch,50)

deltaT = np.linspace(171.02740866424318-10,-0.0024876012556708225+10,50)

def calculateDistance(sim2):
    rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
    marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
    distance = (rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2 +(rocketCord[2]-marsCord[2])**2
    return distance

for i in deltaX:
    for j in deltaY:
        sim=SpaceSim()
        f = sim.simulate([[171.02740866424318,i,j,-5.940960457227606e-05]])
        distance=calculateDistance(f)

        if distance<(0.001):
            x.append(i)
            y.append(j)
            z.append(f.particles[3].m)
        else:
            x1.append(i)
            y1.append(j)
            z1.append(f.particles[3].m)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z,color='r')
ax.scatter(x1, y1, z1,color='b')
ax.set_xlabel('Delta Vx (AU/DAY)')
ax.set_ylabel('Delta Vy (AU/DAY)')
ax.set_zlabel('Objective Function (M_Sun)')


plt.show()
