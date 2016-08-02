from simulatorHigh import SpaceSim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def calculateDistance(sim2):
    rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
    marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
    distance = (rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2 +(rocketCord[2]-marsCord[2])**2
    return distance

deci = [[171.02740866424318, -0.00027481006013446203, -0.0024876012556708225, -5.940960457227606e-05]]

z=[]
x=[]
y=[]

time =171.02740866424318

z1=[]
x1=[]
y1=[]

Patch=0.0009

deltaX=np.linspace(-0.00027481006013446203-Patch,-0.00027481006013446203+Patch,50)
deltaY=np.linspace(-0.0024876012556708225-Patch,-0.0024876012556708225+Patch,50)
deltaT=np.append(np.linspace(171.02740866424318-50,171.02740866424318,100),np.linspace(171.02740866424318,171.02740866424318+50,100))

for idx,t in enumerate(deltaT):
    z=[]
    x=[]
    y=[]

    z1=[]
    x1=[]
    y1=[]
    for i in deltaX:
        for j in deltaY:
            sim=SpaceSim()
            f = sim.simulate([[t,i,j,-5.940960457227606e-05]])
            distance=calculateDistance(f)
            if distance<(0.001):
                x.append(i)
                y.append(j)
                z.append(f.particles[3].m)
            else:
                x1.append(i)
                y1.append(j)
                z1.append(f.particles[3].m)

    print t
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1, y1, z1,color='b')
    ax.scatter(x, y, z,color='r')
    # plt.show()
    plt.savefig('land4/archi%03d' %idx, dpi = 72);
    plt.close()
