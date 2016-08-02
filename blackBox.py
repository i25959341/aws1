from PyGMO.problem import base
import rebound
import numpy as np
import math
from simulator import SpaceSim

MAX_DV= 0.00045512099
filename = "blackBoxLow.bin"

class earthToMars(base):
    def __init__(self, dim=12*3,c_dim=2):
        #USAGE: super(derived_class_name,self).__init__(dim, i_dim, n_obj, c_dim, c_ineq_dim, c_tol)
        #dim: Total dimension of the decision vector
        #i_dim: dimension of the integer part of decision vector (the integer part is placed at the end of the decision vector). Defaults to 0
        #n_obj: number of objectives. Defaults to 1
        #c_dim: total dimension of the constraint vector. dDefaults to 0
        #c_ineq_dim: dimension of the inequality part of the constraint vector (inequality const. are placed at the end of the decision vector). Defaults to 0
        #c_tol: constraint tolerance. When comparing individuals, this tolerance is used to decide whether a constraint is considered satisfied.
        super(earthToMars,self).__init__(dim, 0, 1, c_dim, 0, 1e-5)
        self.set_bounds(-MAX_DV, MAX_DV)
        self.c_dim=c_dim
        self.dim=dim

    def _objfun_impl(self, x):
        sim=SpaceSim()
        dimention =self.dimension/3
        decisions=[]
        de = [0,0,0]
        for i in range(dimention):
            de[0]=x[i*3]
            de[1]=x[i*3+1]
            de[2]=x[i*3+2]
            decisions.append(de)

        f = sim.simulate(decisions)

        return (-f.particles[3].m,)

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
        dimention =self.dimension/3
        decisions=[]
        for i in range(dimention):
            de=[]
            de.append(x[i*3])
            de.append(x[i*3+1])
            de.append(x[i*3+2])
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

    def human_readable_extra(self):
        return "\n\t Problem dimension: " + str(self.__dim)

class earthToMars1(base):
    def __init__(self, dim=12*3,c_dim=2):
        #USAGE: super(derived_class_name,self).__init__(dim, i_dim, n_obj, c_dim, c_ineq_dim, c_tol)
        #dim: Total dimension of the decision vector
        #i_dim: dimension of the integer part of decision vector (the integer part is placed at the end of the decision vector). Defaults to 0
        #n_obj: number of objectives. Defaults to 1
        #c_dim: total dimension of the constraint vector. dDefaults to 0
        #c_ineq_dim: dimension of the inequality part of the constraint vector (inequality const. are placed at the end of the decision vector). Defaults to 0
        #c_tol: constraint tolerance. When comparing individuals, this tolerance is used to decide whether a constraint is considered satisfied.
        super(earthToMars1,self).__init__(dim, 0, 1, c_dim, 0, 1e-5)
        self.set_bounds(-MAX_DV, MAX_DV)
        self.c_dim=c_dim
        self.dim=dim

    def _objfun_impl(self, x):
        sim=SpaceSim()
        dimention =self.dimension/3
        decisions=[]
        de = [0,0,0]
        for i in range(dimention):
            de[0]=x[i*3]
            de[1]=x[i*3+1]
            de[2]=x[i*3+2]
            decisions.append(de)

        f = sim.simulate(decisions)

        return (-f.particles[3].m,)

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
        dimention =self.dimension/3
        decisions=[]
        for i in range(dimention):
            de=[]
            de.append(x[i*3])
            de.append(x[i*3+1])
            de.append(x[i*3+2])
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

    def human_readable_extra(self):
        return "\n\t Problem dimension: " + str(self.__dim)

def run_example1(n_restarts=20):
    from PyGMO import algorithm, island
    prob = earthToMars()
    algo = algorithm.scipy_slsqp(max_iter=500, acc=1e-5)
    algo2 = algorithm.mbh(algo, n_restarts, 0.005)
    algo2.screen_output = True
    isl = island(algo2, prob, 1)
    print("Running Monotonic Basin Hopping .... this will take a while.")
    isl.evolve(1)
    isl.join()
    print("Is the solution found a feasible trajectory? " +
          str(prob.feasibility_x(isl.population.champion.x)))
    return isl.population.champion.x

x=earthToMars()
vector=run_example1()
deci=x._convertToList(vector)
print deci
# sim2=SpaceSim()
# sim=sim2.simulateScatter(deci)
# import rebound
# import matplotlib.pyplot as plt
# fig = rebound.OrbitPlot(sim)
# plt.show()
