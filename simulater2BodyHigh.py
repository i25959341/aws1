import rebound
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

STANDARD_GRAVITY= 9.80665 / 20.0402949
SPECIFIC_IMPLUSE = 350.0 /86400.0

def stopEngine():
    def stop(reb_sim):
        reb_sim.contents.particles[1].ax += 0.0
        reb_sim.contents.particles[1].ay += 0.0
        reb_sim.contents.particles[1].az += 0.0
    return stop

def creatThrust(x,y,z):
    def thrust(reb_sim):
        reb_sim.contents.particles[1].ax += x/reb_sim.contents.dt
        reb_sim.contents.particles[1].ay += y/reb_sim.contents.dt
        reb_sim.contents.particles[1].az += z/reb_sim.contents.dt
    return thrust

class SpaceSim2Body(object):
    def __init__(self,sim):
        self.sim = self._initSim(sim)

    def status(self):
        self.sim.status()

    def _initSim(self,sim2):
        """
        Initialize rebound Simulation with units, integrator, and bodies.
        Code assumes rocket is ps[3] and mars is [2].
        """
        sim = rebound.Simulation()
        sim.units = 'day', 'AU', 'Msun'
        sim.integrator = "ias15"

        mars = sim2.particles[2]
        rocket = sim2.particles[3]

        rocketm = sim2.particles[3].m

        rocketx = rocket.x-mars.x
        rockety = rocket.y-mars.y
        rocketz = rocket.z-mars.z

        rocketvx = rocket.vx-mars.vx
        rocketvy = rocket.vy-mars.vy
        rocketvz = rocket.vz-mars.vz

        sim.add(m=1e-6)
        sim.add(m=rocketm, x=rocketx , y=rockety , z=rocketz,
        vx = rocketvx, vy =rocketvx, vz = rocketvx) #only second stage now
        sim.move_to_com()
        return sim

    def simulate(self,sim2,deci):
        self.sim = self._initSim(sim2)
        A = np.matrix(deci)
        A=A.shape[0]

        dist=[]

        for i in range(A):
            self.sim.integrate(deci[i][0])
            # Start engine
            self.sim.additional_forces = creatThrust(deci[i][1],deci[i][2],
            deci[i][3])
            self.sim.velocity_dependent = 1
            self.sim.integrate(deci[i][0]+1)
            # Stop the engine after a day
            self.sim.additional_forces = stopEngine()
            self.sim.velocity_dependent = 1
            self.loseMass(deci[i][1],deci[i][2],deci[i][3],self.sim)

        self.sim.integrate(400)
        dist.append(self.calculateDistance(self.sim))
        self.sim.integrate(430)
        dist.append(self.calculateDistance(self.sim))
        self.sim.integrate(460)
        dist.append(self.calculateDistance(self.sim))
        self.sim.integrate(490)
        dist.append(self.calculateDistance(self.sim))
        self.sim.integrate(520)
        dist.append(self.calculateDistance(self.sim))

        return (self.sim, dist)

    def simulateScatter(self,deci):
        RESOLUTION=250
        END=520
        A = np.matrix(deci)
        A=A.shape[0]

        xRocket=[]
        yRocket=[]
        zRocket=[]

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
                 xRocket.append(self.sim.particles[1].x)
                 yRocket.append(self.sim.particles[1].y)
                 zRocket.append(self.sim.particles[1].z)

                 xMars.append(self.sim.particles[0].x)
                 yMars.append(self.sim.particles[0].y)
                 zMars.append(self.sim.particles[0].z)

            self.sim.integrate(deci[i][0])
            self.sim.additional_forces = creatThrust(deci[i][1],deci[i][2],
                deci[i][3])
            self.sim.integrate(deci[i][0]+1)
            self.sim.additional_forces = stopEngine()
            self.sim.velocity_dependent = 1

            time= deci[i][0]+2

            xRocket.append(self.sim.particles[1].x)
            yRocket.append(self.sim.particles[1].y)
            zRocket.append(self.sim.particles[1].z)

            xMars.append(self.sim.particles[0].x)
            yMars.append(self.sim.particles[0].y)
            zMars.append(self.sim.particles[0].z)

        times=np.linspace(time,END-1,RESOLUTION)
        for index , time in enumerate(times):
            self.sim.integrate(time)
            xRocket.append(self.sim.particles[1].x)
            yRocket.append(self.sim.particles[1].y)
            zRocket.append(self.sim.particles[1].z)

            xMars.append(self.sim.particles[0].x)
            yMars.append(self.sim.particles[0].y)
            zMars.append(self.sim.particles[0].z)

        self.sim.integrate(END)

        xRocket.append(self.sim.particles[1].x)
        yRocket.append(self.sim.particles[1].y)
        zRocket.append(self.sim.particles[1].z)

        xMars.append(self.sim.particles[0].x)
        yMars.append(self.sim.particles[0].y)
        zMars.append(self.sim.particles[0].z)


        ax.scatter(xRocket, yRocket, zRocket)
        ax.plot(xRocket,yRocket,zRocket, label='Rocket')
        ax.scatter(xMars, yMars, zMars)
        ax.plot(xMars, yMars, zMars, label='Mars')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        plt.show()
        return xRocket, yRocket, zRocket , xMars, yMars, zMars

    def loseMass(self,x,y,z,reb_sim):
            deltaV = ((x)**2.0+(y)**2.0+(z)**2.0)**0.5
            reb_sim.particles[1].m = reb_sim.particles[1].m \
            *math.exp(-deltaV/(SPECIFIC_IMPLUSE*STANDARD_GRAVITY))  # Mass loss

    def calculateDistance(self,sim2):
        rocketCord = (sim2.particles[1].x,sim2.particles[1].y,sim2.particles[1].z)
        marsCord = (sim2.particles[0].x,sim2.particles[0].y,sim2.particles[0].z)
        distance = ((rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2 \
        +(rocketCord[2]-marsCord[2])**2)**0.5

        return distance

    def animate(self, parameters):
        xRocket, yRocket, zRocket , xMars, yMars, zMars = self.simulateScatter(parameters)
        N_trajectories=2

        rocket = zip (xRocket, yRocket, zRocket)
        mar = zip(xMars, yMars, zMars)

        x_t = np.array([rocket,mar])

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
        ax.set_xlim((min(xRocket), max(xRocket)))
        ax.set_ylim((min(yRocket), max(yRocket)))
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

        # anim.save('marsToEarth.gif', writer='imagemagick', fps=30)
        plt.show()
