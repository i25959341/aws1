import rebound
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time as clock

number = 0

STANDARD_GRAVITY= 9.80665 / 20.0402949
SPECIFIC_IMPLUSE = 6000.0 /86400.0

def stopEngine(reb_sim):
    reb_sim.contents.particles[3].ax += 0.0
    reb_sim.contents.particles[3].ay += 0.0
    reb_sim.contents.particles[3].az += 0.0

def creatThrust(x,y,z):
    def thrust(reb_sim):
        ax=x*reb_sim.contents.dt/30.0
        ay=y*reb_sim.contents.dt/30.0
        az=z*reb_sim.contents.dt/30.0

        reb_sim.contents.particles[3].ax += ax/reb_sim.contents.dt
        reb_sim.contents.particles[3].ay += ay/reb_sim.contents.dt
        reb_sim.contents.particles[3].az += az/reb_sim.contents.dt
    return thrust

class SpaceSim(object):
    def __init__(self):
        self.sim = self._initSim()

    def status(self):
        self.sim.status()

    def _initSim(self):
        """
        Initialize rebound Simulation with units, integrator, and bodies.
        Code assumes rocket is ps[3] and mars is [2].
        """
        sim = rebound.Simulation()
        sim.units = 'day', 'AU', 'Msun'
        sim.integrator = "ias15"

        sim.add(m=1.)
        sim.add(m=1e-6, a=1.)
        sim.add(m=1e-6, a=1.5, Omega =3.0)
        sim.add(m=4.13E-27, a=1.01) #only second stage now

        sim.move_to_com()
        return sim

    def loseMass(self,x,y,z,reb_sim):
            deltaV = ((x)**2.0+(y)**2.0+(z)**2.0)**0.5
            reb_sim.particles[3].m = reb_sim.particles[3].m \
            *math.exp(-deltaV/(SPECIFIC_IMPLUSE*STANDARD_GRAVITY))  # Mass loss

    def simulate(self,parameters):
        A = np.matrix(parameters)
        A=A.shape
        times=np.linspace(0,360,A[0])
        t0 = clock.time()
        for i , time in enumerate(times):
            decision=parameters[i]
            self.sim.additional_forces = creatThrust(decision[0],decision[1],decision[2])
            self.sim.velocity_dependent = 1
            self.sim.integrate(time)
            self.loseMass(decision[0],decision[1],decision[2],self.sim)
        t1 = clock.time()
        total =t1-t0
        if total>0.9:
            print "time 1 is " + str(total)
            print "Deci is " + str(parameters)
        return self.sim

    def simulate1(self,parameters,sim):
        self.sim=sim
        A = np.matrix(parameters)
        A=A.shape
        times=np.linspace(sim.t,sim.t+30,A[0])

        for i , time in enumerate(times):
            decision=parameters[i]
            self.sim.additional_forces = creatThrust(decision[0],decision[1],decision[2])
            self.sim.velocity_dependent = 1
            self.sim.integrate(time)
            self.loseMass(decision[0],decision[1],decision[2],self.sim)
        return self.sim

    def simulateGraph(self,parameters):
        A = np.matrix(parameters)
        A=A.shape
        times=np.linspace(0,360,A[0])

        for i , time in enumerate(times):
            decision=parameters[i]
            self.sim.additional_forces = creatThrust(decision[0],decision[1],decision[2])
            self.sim.velocity_dependent = 1
            self.sim.integrate(time)
            self.loseMass(decision[0],decision[1],decision[2],self.sim)
        return self.sim

    def simulateScatter(self,parameters):
            A = np.matrix(parameters)
            A=A.shape
            resolution=15
            times=np.linspace(0,360,A[0])
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
            for i , time in enumerate(times):
                decision=parameters[i]
                self.sim.additional_forces = creatThrust(decision[0],decision[1],decision[2])
                self.sim.velocity_dependent = 1

                times2=np.linspace(self.sim.t,time-1,resolution)
                for j , time2 in enumerate(times2):
                    self.sim.integrate(time2)
                    xRocket.append(self.sim.particles[3].x)
                    yRocket.append(self.sim.particles[3].y)
                    zRocket.append(self.sim.particles[3].z)

                    xEarth.append(self.sim.particles[1].x)
                    yEarth.append(self.sim.particles[1].y)
                    zEarth.append(self.sim.particles[1].z)

                    xMars.append(self.sim.particles[2].x)
                    yMars.append(self.sim.particles[2].y)
                    zMars.append(self.sim.particles[2].z)

                self.sim.integrate(time)
                self.loseMass(decision[0],decision[1],decision[2],self.sim)
                # print self.sim.particles[3].z
                xRocket.append(self.sim.particles[3].x)
                yRocket.append(self.sim.particles[3].y)
                zRocket.append(self.sim.particles[3].z)

                xEarth.append(self.sim.particles[1].x)
                yEarth.append(self.sim.particles[1].y)
                zEarth.append(self.sim.particles[1].z)

                xMars.append(self.sim.particles[2].x)
                yMars.append(self.sim.particles[2].y)
                zMars.append(self.sim.particles[2].z)

            times=np.linspace(self.sim.t,self.sim.t+100,50)
            for i , time in enumerate(times):
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

            # A = np.matrix(parameters1)
            # A=A.shape
            # times=np.linspace(self.sim.t,self.sim.t+30,A[0])
            # for i , time in enumerate(times):
            #     decision=parameters1[i]
            #     self.sim.additional_forces = creatThrust(decision[0],decision[1],decision[2])
            #     self.sim.velocity_dependent = 1
            #     times2=np.linspace(self.sim.t,time-1,resolution)
            #
            #     for j , time2 in enumerate(times2):
            #         self.sim.integrate(time2)
            #         xRocket.append(self.sim.particles[3].x)
            #         yRocket.append(self.sim.particles[3].y)
            #         zRocket.append(self.sim.particles[3].z)
            #
            #         xEarth.append(self.sim.particles[1].x)
            #         yEarth.append(self.sim.particles[1].y)
            #         zEarth.append(self.sim.particles[1].z)
            #
            #         xMars.append(self.sim.particles[2].x)
            #         yMars.append(self.sim.particles[2].y)
            #         zMars.append(self.sim.particles[2].z)
            #
            #     self.sim.integrate(time)
            #     self.loseMass(decision[0],decision[1],decision[2],self.sim)
            #
            # times=np.linspace(self.sim.t,self.sim.t+30,resolution)
            # for i , time in enumerate(times):
            #     self.sim.integrate(time)
            #     xRocket.append(self.sim.particles[3].x)
            #     yRocket.append(self.sim.particles[3].y)
            #     zRocket.append(self.sim.particles[3].z)
            #
            #     xEarth.append(self.sim.particles[1].x)
            #     yEarth.append(self.sim.particles[1].y)
            #     zEarth.append(self.sim.particles[1].z)
            #
            #     xMars.append(self.sim.particles[2].x)
            #     yMars.append(self.sim.particles[2].y)
            #     zMars.append(self.sim.particles[2].z)


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

# def creatStage():
#     def stage(reb_sim):
#         ax=reb_sim.contents.particles[3].ax
#         ay=reb_sim.contents.particles[3].ay
#         az=reb_sim.contents.particles[3].az
#
#         m=reb_sim.contents.particles[3].m
#
#         a2=(ax**2+ay**2+az**2)**0.5
#
#         unitVector = [ax/a2,ay/a2,az/a2]
#
#         vE = 0.00005
#
#         deltaV = vE*np.log(1/0.9)
#
#         ax=deltaV*unitVector[0]
#         ay=deltaV*unitVector[1]
#         az=deltaV*unitVector[2]
#
#         reb_sim.contents.particles[3].ax += ax
#         reb_sim.contents.particles[3].ay += ay
#         reb_sim.contents.particles[3].az += az
#
#         reb_sim.contents.particles[3].m = reb_sim.contents.particles[3].m*0.9  # assume 10% mass loss
#     return stage
# def generateSolutions():
#     decision = [0.,0.,0.,False]
#     decisions=[]
#     THRUST=0.00005
#     for i in range(361):
#         decision[0]=np.random.uniform(-THRUST,THRUST)
#         decision[1]=np.random.uniform(-THRUST,THRUST)
#         decision[2]=np.random.uniform(-THRUST,THRUST)
#         decision[3]=np.random.randint(2)
#         decisions.append(decision)
#     x = SpaceSim()
#     k=x.simulate(decisions)
#     # print k.particles[3].m
#     fig = rebound.OrbitPlot(k)
#     plt.show()
#
# generateSolutions()
