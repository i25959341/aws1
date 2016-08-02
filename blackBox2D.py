from PyGMO.problem import base
import rebound
import numpy as np
import math
from simulator import SpaceSim

MAX_DV= 0.00045512099
filename = "blackBoxLow2D.bin"

class earthToMars2D(base):
    def __init__(self, dim=12*2,c_dim=2):
        super(earthToMars2D,self).__init__(dim, 0, 1, c_dim, 0, 1e-5)
        self.set_bounds(-MAX_DV, MAX_DV)
        self.c_dim = c_dim
        self.dim = dim

    def _objfun_impl(self, x):
        sim=SpaceSim()
        decisions=self._convertToList(x)
        f = sim.simulate(decisions)
        return (-f.particles[3].m, )

    def _compute_constraints_impl(self, x):
        sim=SpaceSim()
        decisions=self._convertToList(x)
        sim2 = sim.simulate(decisions)

        dvx,dvy,dvz = self.getRelativeVelocity(sim2)
        distance = self.getDistance(sim2)

        ceq = list([0]*self.c_dim)

        ceq[0]=distance-8e-6 # distance-0.01<=0
        ceq[1]=(dvx**2+dvy**2+dvz**2)**0.5-0.0004
        return ceq  # distance-0.01<=0

    def _convertToList(self,x):
        dimention =self.dimension/2
        decisions=[]
        for i in range(dimention):
            de = []
            de.append(x[i*2])
            de.append(x[i*2+1])
            de.append(0)
            decisions.append(de)
        return decisions

    def getDistance(self, sim2):
        rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
        marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
        distance = ((rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2 +
        (rocketCord[2]-marsCord[2])**2)**0.5

        return distance
    def getRelativeVelocity(self, sim2):
        rocketCord = (sim2.particles[3].vx,sim2.particles[3].vy,sim2.particles[3].vz)
        marsCord = (sim2.particles[2].vx,sim2.particles[2].vy,sim2.particles[2].vz)
        return rocketCord[0]-marsCord[0],rocketCord[1]-marsCord[1],rocketCord[2]-marsCord[2]

MAX_DV2= 0.00045512099

class rocketMars2D(base):
    def __init__(self, dim=3*2,c_dim=1):
        super(rocketMars2D,self).__init__(dim, 0, 1, c_dim, 0, 1e-4)
        self.set_bounds(-MAX_DV2, MAX_DV2)
        self.c_dim=c_dim
        self.dim=dim

    def _objfun_impl(self, x):
        sim2 = rebound.Simulation.from_file(filename)
        sim=SpaceSim()
        decisions=self._convertToList(x)
        f = sim.simulate1(decisions,sim2)
        dvx,dvy,dvz = self.getRelativeVelocity(f)
        dv= (dvx**2+dvy**2+dvz**2)**0.5
        return (dv,)

    def _compute_constraints_impl(self, x):
        sim2 = rebound.Simulation.from_file(filename)
        sim=SpaceSim()
        decisions = self._convertToList(x)
        sim2 = sim.simulate1(decisions,sim2)

        distance = self.getDistance(sim2)
        dvx, dvy, dvz = self.getRelativeVelocity(sim2)

        ceq = list([0]*self.c_dim)
        ceq[0]=distance-8e-3
        return ceq  # distance-0.01<=0

    def _convertToList(self,x):
        dimention =self.dimension/2
        decisions=[]
        for i in range(dimention):
            de = []
            de.append(x[i*2])
            de.append(x[i*2+1])
            de.append(0)
            decisions.append(de)
        return decisions

    def getDistance(self, sim2):
        rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
        marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
        distance = ((rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2 +
        (rocketCord[2]-marsCord[2])**2)**0.5
        return distance

    def getRelativeVelocity(self, sim2):
        rocketCord = (sim2.particles[3].vx,sim2.particles[3].vy,sim2.particles[3].vz)
        marsCord = (sim2.particles[2].vx,sim2.particles[2].vy,sim2.particles[2].vz)
        return rocketCord[0]-marsCord[0],rocketCord[1]-marsCord[1],rocketCord[2]-marsCord[2]

def run_example1(n_restarts=5):
    from PyGMO import algorithm, island
    prob = earthToMars2D()
    algo = algorithm.scipy_slsqp(max_iter=500, acc=1e-5)
    # algo.screen_output = True
    algo2 = algorithm.mbh(algo, 5, 0.05)
    algo2.screen_output = True
    isl = island(algo2, prob, 1)
    print("Running Monotonic Basin Hopping .... this will take a while.")
    isl.evolve(1)
    isl.join()
    print("Is the solution found a feasible trajectory? " +
          str(prob.feasibility_x(isl.population.champion.x)))
    return isl.population.champion.x

def run_example2(n_restarts=5):
    from PyGMO import algorithm, island
    prob = rocketMars2D()
    algo = algorithm.scipy_slsqp(max_iter=500, acc=1e-5)
    # algo.screen_output = True
    algo2 = algorithm.mbh(algo, 5, 0.05)
    algo2.screen_output = True
    isl = island(algo2, prob, 1)
    print("Running Monotonic Basin Hopping .... this will take a while.")
    isl.evolve(1)
    isl.join()
    print("Is the solution found a feasible trajectory? " +
          str(prob.feasibility_x(isl.population.champion.x)))
    return isl.population.champion.x

def run_example3(n_restarts=5):
    from PyGMO import algorithm, island
    prob = earthToMars2D()
    algo = algorithm.scipy_slsqp(max_iter=500, acc=1e-5)
    # algo.screen_output = True
    algo2 = algorithm.mbh(algo, 5, 0.05)
    algo2.screen_output = True
    pop = population(prob)
    pop.push_back((-3.810404036178629e-05, 8.036667366434322e-05, 0, -0.00011631957922519811, -0.0003960700040113729, 0, 0.00014343900668268246, -0.00039460589468829016, 0, 0.0004133243825183847, -0.0002676479632615287, 0, -6.353773946676955e-05, -0.0004027302771161609, 0, -0.00019483461157664088, -0.0003938299142410649, 0, 0.0003740376551173652, -0.00045439735580127933, 0, 0.00026271994456226056, -4.17413726080276e-05, 0, 0.0004025294016016401, 9.22186764465555e-05, 0, 0.0004379362102351141, -8.202101747983173e-05, 0, 2.0842990495214604e-05, -1.927554372930426e-05, 0, -2.392388475139966e-05, -6.3420840462436174e-06, 0))
    pop.push_back((0.00018354551497353738, 0.0002897005581203533, 0, 9.385407683672441e-05, -0.0004375546286935724, 0, -0.00017406053466786356, -0.0004055793819144533, 0, 7.811816626063441e-05, -0.00028869842254392053, 0, 0.000280132941671916, -0.00045467528344872834, 0, 0.00031161406626870487, -0.0004418005074233615, 0, 0.00016912620000403375, -0.00045156036938030775, 0, 0.00043500734938167605, -4.4611940286304056e-05, 0, 0.00023373694896547512, 4.622353180355802e-06, 0, 0.00043504614537196785, -0.00042017445674379463, 0, 0.00016822207354911628, 0.00010574669088542543, 0, 2.1129649656070842e-05, 0.00020199652091584146, 0))

    isl = island(algo2,pop)
    print("Running Monotonic Basin Hopping .... this will take a while.")
    isl.evolve(1)
    isl.join()
    print("Is the solution found a feasible trajectory? " +
          str(prob.feasibility_x(isl.population.champion.x)))
    return isl.population.champion.x
 # **deci = [[0.00018354551497353738, 0.0002897005581203533, 0], [9.385407683672441e-05, -0.0004375546286935724, 0], [-0.00017406053466786356, -0.0004055793819144533, 0], [7.811816626063441e-05, -0.00028869842254392053, 0], [0.000280132941671916, -0.00045467528344872834, 0], [0.00031161406626870487, -0.0004418005074233615, 0], [0.00016912620000403375, -0.00045156036938030775, 0], [0.00043500734938167605, -4.4611940286304056e-05, 0], [0.00023373694896547512, 4.622353180355802e-06, 0], [0.00043504614537196785, -0.00042017445674379463, 0], [0.00016822207354911628, 0.00010574669088542543, 0], [2.1129649656070842e-05, 0.00020199652091584146, 0]]

#
x=earthToMars2D()
# y=rocketMars2D()
# vector=run_example1()
# deci=x._convertToList(vector)
# print deci
# sim2=SpaceSim()
# sim=sim2.simulate(deci)
# sim.save(filename)
# sim2=SpaceSim()
# sim2.simulateScatter(deci)
#
vector=run_example1()
deci=x._convertToList(vector)
print deci
