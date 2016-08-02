from PyGMO.problem import base
import rebound
import numpy as np
import math
from simulatorStaging import SpaceSim

class earthToMars(base):
    def __init__(self, dim=12*4,c_dim=1):
        super(earthToMars,self).__init__(dim, dim/4, 1, 1, 0, 1e-6)
        # USAGE: super(derived_class_name,self).__init__(dim, i_dim, n_obj, c_dim, c_ineq_dim, c_tol)
        self.set_bounds([-0.0005]*(dim-dim/4)+[0]*(dim/4),[0.0005]*(dim-dim/4)+[1]*(dim/4) )
    def _objfun_impl(self, x):
        sim=SpaceSim()
        decisions=self._convertToList(x)
        f = sim.simulate(decisions)
        return (f.particles[3].m, )

    def _compute_constraints_impl(self, x):
        sim=SpaceSim()
        decisions=self._convertToList(x)

        sim2 = sim.simulate(decisions)
        rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
        marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
        distance = (rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2 +(rocketCord[2]-marsCord[2])**2
        ceq = list([0]*1)
        ceq[0]=distance-0.01 # distance-0.01<=0
        return ceq  # distance-0.01<=0

    def _convertToList(self,x):
        dimention1 = self.dimension- self.dimension/4
        dimention =dimention1/3
        decisions=[]
        for i in range(dimention):
            de=[]
            de.append(x[i*3])
            de.append(x[i*3+1])
            de.append(x[i*3+2])
            de.append(False)
            decisions.append(de)

        for i in range(self.dimension/4):
            decisions[i][3]=x[dimention1]
            dimention1=dimention1+1

        return decisions

    def _compare_fitness_impl(self, f1, f2):
        return f1[0] > f2[0]

    def human_readable_extra(self):
        return "\n\t Problem dimension: " + str(self.__dim)

def run_example1(n_restarts=5):
    from PyGMO import algorithm, island
    prob = earthToMars()
    algo = algorithm.scipy_slsqp(max_iter=500, acc=1e-5)
    algo2 = algorithm.mbh(algo, 5, 0.05)
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
sim2=SpaceSim()
sim2.simulateGraph(deci)
