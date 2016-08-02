import rebound
import numpy as np
import math
import matplotlib.pyplot as plt

def creatThrust(x,y,z):
    def thrust(reb_sim):
        reb_sim.contents.particles[3].ax += x
        reb_sim.contents.particles[3].ay += y
        reb_sim.contents.particles[3].az += z

        deltaV = (reb_sim.contents.particles[3].ax**2.0+reb_sim.contents.particles[3].ay**2.0+reb_sim.contents.particles[3].az**2.0)**0.5
        reb_sim.contents.particles[3].m = reb_sim.contents.particles[3].m*math.exp(-deltaV/10)  # Mass loss
    return thrust

class SpaceSim(object):
    def __init__(self):
        self.sim = self._initSim()

    def status(self):
        self.sim.status()

    def make_move(self, x,y,z,t):
        rocket = self.sim.particles[3]

        self.sim.additional_forces = accelerateDown
        self.sim.integrate(self.sim.t + 1)  # time + 1 hr

        return self.get_reward()

    def setXYZ(self,x,y,z):
        THRUSTX=x
        THRUSTY=y
        THRUSTZ=z

    def _initSim(self):
        """
        Initialize rebound Simulation with units, integrator, and bodies.
        Code assumes rocket is ps[3] and mars is [2].
        """
        sim = rebound.Simulation()
        sim.units = 'day', 'AU', 'Msun'
        sim.integrator = "ias15"

        # add planets yourself.
        sim.add(id=2, m=1.)
        sim.add(id=5, m=1e-6, a=1.)
        sim.add(m=1e-6, a=2., Omega =3.0)
        sim.add(m=1e-34, a=1.01)

        sim.move_to_com()
        return sim

    def simulate(self,parameters):
        A = np.matrix(parameters)
        A=A.shape
        times=np.linspace(0,A[0]-1,A[0])
        for i , time in enumerate(times):
            decision=parameters[i]
            self.setXYZ(decision[0],decision[1],decision[2])
            self.sim.additional_forces = creatThrust(decision[0],decision[1],decision[2])
            self.sim.velocity_dependent = 1
            self.sim.integrate(time)

        return self.sim

    def plot(self):
        fig = rebound.OrbitPlot(self.sim)


def stopEngine(reb_sim):
    reb_sim.contents.particles[3].ax += 0.0
    reb_sim.contents.particles[3].ay += 0.0
    reb_sim.contents.particles[3].az += 0.0

decision = [0,0,0,False]

decisions=[]

for i in range(361):
	decisions.append([i]+decision)

def generateSolutions():
    decision = [0.,0.,0.,False]
    decisions=[]
    THRUST=0.0005
    for i in range(361):
        decision[0]=np.random.uniform(-THRUST,THRUST)
        decision[1]=np.random.uniform(-THRUST,THRUST)
        decision[2]=np.random.uniform(-THRUST,THRUST)
        decisions.append(decision)
    x = SpaceSim()
    k=x.simulate(decisions)
    print k.particles[3].m
# generateSolutions()
