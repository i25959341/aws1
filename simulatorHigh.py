import rebound
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time

STANDARD_GRAVITY= 9.80665 / 20.0402949
SPECIFIC_IMPLUSE = 350.0 /86400.0

def stopEngine():
    def stop(reb_sim):
        reb_sim.contents.particles[3].ax += 0.0
        reb_sim.contents.particles[3].ay += 0.0
        reb_sim.contents.particles[3].az += 0.0
    return stop

def creatThrust(x,y,z):
    def thrust(reb_sim):
        reb_sim.contents.particles[3].ax += x/reb_sim.contents.dt
        reb_sim.contents.particles[3].ay += y/reb_sim.contents.dt
        reb_sim.contents.particles[3].az += z/reb_sim.contents.dt
    return thrust

def creatThrust2(x,y,z):
    def thrust(reb_sim):
        ax=x*reb_sim.contents.dt/1.0
        ay=y*reb_sim.contents.dt/1.0
        az=z*reb_sim.contents.dt/1.0

        reb_sim.contents.particles[3].ax += ax/reb_sim.contents.dt
        reb_sim.contents.particles[3].ay += ay/reb_sim.contents.dt
        reb_sim.contents.particles[3].az += az/reb_sim.contents.dt
    return thrust

class SpaceSim(object):
    def __init__(self,sim2="none",wh=False):
        if wh==True:
            self.sim = self._initSim(wh=True)
            self.wh=True
        elif sim2!="none":
            self.sim = sim2
            self.wh=False
        else:
            self.sim = self._initSim()
            self.wh=False

    def status(self):
        self.sim.status()

    def _initSim(self,wh=False):
        """
        Initialize rebound Simulation with units, integrator, and bodies.
        Code assumes rocket is ps[3] and mars is [2].
        """
        if wh==False:
            sim = rebound.Simulation()
            sim.units = 'day', 'AU', 'Msun'
            sim.integrator = "ias15"
        else:
            sim = rebound.Simulation()
            sim.units = 'day', 'AU', 'Msun'
            sim.integrator = "whfast"
            sim.gravity='basic'
            sim.dt = 1e-1
        # add planets yourself.
        sim.add(m=1.)
        sim.add(m=1e-6, a=1.)
        sim.add(m=1e-6, a=1.5, Omega =3.0)
        sim.add(m=4.13E-27, a=1.01) #only second stage now

        sim.move_to_com()
        return sim

    def getMassFinal(self,deci):
        sim = self.simulate(deci)
        return sim.particles[3].m

    def simulate(self,deci):
        t0 = time.time()
        self.sim = self._initSim()
        A = np.matrix(deci)
        A=A.shape[0]
        t00 = time.time()
        for i in range(A):

            self.sim.integrate(deci[i][0])
            # Start engine
            if self.wh==False:
                self.sim.additional_forces = creatThrust(deci[i][1],deci[i][2],
                deci[i][3])
            else:
                self.sim.additional_forces = creatThrust2(deci[i][1],deci[i][2],
                deci[i][3])

            self.sim.velocity_dependent = 1
            self.sim.integrate(deci[i][0]+1)
            # Stop the engine after a day
            self.sim.additional_forces = stopEngine()
            self.sim.velocity_dependent = 1
            self.loseMass(deci[i][1],deci[i][2],deci[i][3],self.sim)

        t11 = time.time()
        total1 = t11-t00
        if total1>0.9:
            print "time 1 is " + str(total1)
            print "Deci is " + str(deci)
        self.sim.integrate(480)

        t1 = time.time()
        total = t1-t0

        #check if there is any problem on simulation time
        if total>0.9:
            print "time 2 is " + str(total)
            print "Deci is " + str(deci)

        return self.sim

    def simulate2(self,deci):
        t0 = time.time()
        self.sim = self._initSim()
        A = np.matrix(deci)
        A=A.shape[0]
        t00 = time.time()
        for i in range(A):
            self.sim.integrate(deci[i][0])
            # Start engine
            if self.wh==False:
                self.sim.additional_forces = creatThrust(deci[i][1],deci[i][2],
                deci[i][3])
            else:
                self.sim.additional_forces = creatThrust2(deci[i][1],deci[i][2],
                deci[i][3])
            self.sim.velocity_dependent = 1
            self.sim.integrate(deci[i][0]+1)
            # Stop the engine after a day
            self.sim.additional_forces = stopEngine()
            self.sim.velocity_dependent = 1
            self.loseMass(deci[i][1],deci[i][2],deci[i][3],self.sim)

        t11 = time.time()
        total1 = t11-t00
        if total1>0.9:
            print "time 1 is " + str(total1)
            print "Deci is " + str(deci)
        self.sim.integrate(deci[0][3])

        t1 = time.time()

        total = t1-t0

        #check if there is any problem on simulation time
        if total>0.9:
            print "time 2 is " + str(total)
            print "Deci is " + str(deci)

        return self.sim

    def simulate1(self,deci,sim):
        self.sim = sim
        A = np.matrix(deci)
        A=A.shape[0]

        t0 = time.time()

        for i in range(A):
            self.sim.integrate(deci[i][0])
            # Start engine
            if self.wh==False:
                self.sim.additional_forces = creatThrust(deci[i][1],deci[i][2],
                deci[i][3])
            else:
                self.sim.additional_forces = creatThrust2(deci[i][1],deci[i][2],
                deci[i][3])
            self.sim.velocity_dependent = 1
            self.sim.integrate(deci[i][0]+1)
            # Stop the engine after a day
            self.sim.additional_forces = stopEngine()
            self.sim.velocity_dependent = 1
            self.loseMass(deci[i][1],deci[i][2],deci[i][3],self.sim)

        t1 = time.time()
        total = t1-t0
        if total>0.9:
            print "time 1 is " + str(total)
            print "Deci is " + str(deci)

        t0 = time.time()
        self.sim.integrate(501.0)
        t1 = time.time()
        total = t1-t0
        if total>0.9:
            print "time 1 is " + str(total)
            print "Deci is " + str(deci)
        return self.sim

    def getRelativeVelocity(self, sim2):
        rocketCord = (sim2.particles[3].vx,sim2.particles[3].vy,sim2.particles[3].vz)
        marsCord = (sim2.particles[2].vx,sim2.particles[2].vy,sim2.particles[2].vz)
        return rocketCord[0]-marsCord[0],rocketCord[1]-marsCord[1],rocketCord[2]-marsCord[2]

    def integrate(self,sim, stop):
        self.sim = sim
        self.sim.integrate(stop)
        return self.sim

    def calculateDistance(self,sim2):
        rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
        marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
        distance = ((rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2
        +(rocketCord[2]-marsCord[2])**2)**0.5

        return distance

    def simulateRefine(self,deci,step):
        self.sim.dt=step

        A = np.matrix(deci)
        A=A.shape[0]

        time=0

        for i in range(A):
            for t in np.arange(time,int(deci[i][0]),step):
                self.sim.integrate(t,exact_finish_time=0)

            self.sim.integrate(deci[i][0],exact_finish_time=0)
            # Start engine
            if self.wh==False:
                self.sim.additional_forces = creatThrust(deci[i][1],deci[i][2],
                deci[i][3])
            else:
                self.sim.additional_forces = creatThrust2(deci[i][1],deci[i][2],
                deci[i][3])
            self.sim.velocity_dependent = 1
            self.sim.integrate(deci[i][0]+1,exact_finish_time=0)
            # Stop the engine after a day
            self.sim.additional_forces = stopEngine()
            self.loseMass(deci[i][1],deci[i][2],deci[i][3],self.sim)
            self.sim.velocity_dependent = 1

            time = int(deci[i][0]+2)

        for t in np.arange(time,480,step):
            self.sim.integrate(t,exact_finish_time=0)

        self.sim.integrate(480,exact_finish_time=0)
        # self.sim.status()

        return self.sim

    def loseMass(self,x,y,z,reb_sim):
            deltaV = ((x)**2.0+(y)**2.0+(z)**2.0)**0.5
            reb_sim.particles[3].m = reb_sim.particles[3].m \
            *math.exp(-deltaV/(SPECIFIC_IMPLUSE*STANDARD_GRAVITY))  # Mass loss

    def simulateScatter(self,deci):
            RESOLUTION=50
            END=980
            A = np.matrix(deci)
            A=A.shape[0]

            xRocket=[]
            yRocket=[]
            zRocket=[]

            xEarth=[]
            yEarth=[]
            zEarth=[]

            xMars=[]
            yMars=[]
            zMars=[]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            time = 0
            for i in range(A):
                times=np.linspace(time,deci[i][0]-1,RESOLUTION)
                for index , time in enumerate(times):
                     self.sim.integrate(time)
                     xRocket.append(self.sim.particles[3].x)
                     yRocket.append(self.sim.particles[3].y)
                     zRocket.append(self.sim.particles[3].z)

                     xEarth.append(self.sim.particles[1].x)
                     yEarth.append(self.sim.particles[1].y)
                     zEarth.append(self.sim.particles[1].z)

                     xMars.append(self.sim.particles[2].x)
                     yMars.append(self.sim.particles[2].y)
                     zMars.append(self.sim.particles[2].z)

                self.sim.integrate(deci[i][0])
                if self.wh==False:
                    self.sim.additional_forces = creatThrust(deci[i][1],deci[i][2],
                    deci[i][3])
                else:
                    self.sim.additional_forces = creatThrust2(deci[i][1],deci[i][2],
                    deci[i][3])
                self.sim.integrate(deci[i][0]+1)
                self.sim.additional_forces = stopEngine()
                self.sim.velocity_dependent = 1

                time= deci[i][0]+2

                xRocket.append(self.sim.particles[3].x)
                yRocket.append(self.sim.particles[3].y)
                zRocket.append(self.sim.particles[3].z)

                xEarth.append(self.sim.particles[1].x)
                yEarth.append(self.sim.particles[1].y)
                zEarth.append(self.sim.particles[1].z)

                xMars.append(self.sim.particles[2].x)
                yMars.append(self.sim.particles[2].y)
                zMars.append(self.sim.particles[2].z)

            times=np.linspace(time,END-1,RESOLUTION)
            for index , time in enumerate(times):
                     self.sim.integrate(time)
                     xRocket.append(self.sim.particles[3].x)
                     yRocket.append(self.sim.particles[3].y)
                     zRocket.append(self.sim.particles[3].z)

                     xEarth.append(self.sim.particles[1].x)
                     yEarth.append(self.sim.particles[1].y)
                     zEarth.append(self.sim.particles[1].z)

                     xMars.append(self.sim.particles[2].x)
                     yMars.append(self.sim.particles[2].y)
                     zMars.append(self.sim.particles[2].z)

            self.sim.integrate(END)

            xRocket.append(self.sim.particles[3].x)
            yRocket.append(self.sim.particles[3].y)
            zRocket.append(self.sim.particles[3].z)

            xEarth.append(self.sim.particles[1].x)
            yEarth.append(self.sim.particles[1].y)
            zEarth.append(self.sim.particles[1].z)

            xMars.append(self.sim.particles[2].x)
            yMars.append(self.sim.particles[2].y)
            zMars.append(self.sim.particles[2].z)

            ax.set_xlabel('x (AU)')
            ax.set_ylabel('y (AU)')
            ax.set_zlabel('z (AU)')

            plt.title('Rocket Traveling from HEO to LMO')

            ax.scatter(xRocket, yRocket, zRocket)
            ax.plot(xRocket,yRocket,zRocket, label='Rocket')

            ax.scatter(xEarth, yEarth, zEarth)
            ax.plot(xEarth, yEarth, zEarth, label='Earth')

            ax.scatter(xMars, yMars, zMars)
            ax.plot(xMars, yMars, zMars, label='Mars')

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)

            plt.show()

            return xRocket, yRocket, zRocket , xEarth, \
            yEarth, zEarth, xMars, yMars, zMars

    def animate(self, parameters):
        xRocket, yRocket, zRocket , xEarth, yEarth, zEarth, xMars, yMars, zMars = self.simulateScatter(parameters)
        N_trajectories=3

        rocket = zip (xRocket, yRocket, zRocket)
        earth = zip(xEarth, yEarth, zEarth)
        mar = zip(xMars, yMars, zMars)

        x_t = np.array([rocket,earth,mar])

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        # ax.axis('off')

        # choose a different color for each trajectory
        colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

        lines = sum([ax.plot([], [], [], '-', c=c)
                     for c in colors], [])
        pts = sum([ax.plot([], [], [], 'o', c=c)
                   for c in colors], [])

        # prepare the axes limits
        ax.set_xlim((min(xMars), max(xMars)))
        ax.set_ylim((min(yMars), max(yMars)))
        ax.set_zlim((min(zRocket), max(zRocket)))

        # set point-of-view: specified by (altitude degrees, azimuth degrees)
        ax.view_init(30, 0)

        def init():
            for line, pt in zip(lines, pts):
                line.set_data([], [])
                line.set_3d_properties([])

                pt.set_data([], [])
                pt.set_3d_properties([])
            return lines + pts

        def animate(i):
            # we'll step two time-steps per frame.  This leads to nice results.
            i = (1 * i) % x_t.shape[1]

            for line, pt, xi in zip(lines, pts, x_t):
                x, y, z = xi[:i].T
                line.set_data(x, y)
                line.set_3d_properties(z)

                pt.set_data(x[-1:], y[-1:])
                pt.set_3d_properties(z[-1:])

            ax.view_init(30, 0.3 * i)
            fig.canvas.draw()
            return lines + pts

        # instantiate the animator.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=5000, interval=30, blit=True)

        # anim.save('marsToEarthCapture2D.gif', writer='imagemagick', fps=30)
        plt.show()

    def plot(self):
        fig = rebound.OrbitPlot(self.sim)


def generateSolutions():
   decision = [60.,0.005,0.005,0.005]
   decisions=[]
   THRUST=0.01166964393
   for i in range(1):
       decision[0]=np.random.uniform(300,480)
       decision[1]=np.random.uniform(-THRUST,THRUST)
       decision[2]=np.random.uniform(-THRUST,THRUST)
       decision[2]=np.random.uniform(-THRUST,THRUST)
       decisions.append(decision)
   x = SpaceSim()
   k=x.simulate(decisions)
   # print k.particles[3].m
   fig = rebound.OrbitPlot(k)
   plt.show()

# generateSolutions()
