from PyGMO.problem import base
import rebound
import numpy as np
import math
from simulatorHigh import SpaceSim
import matplotlib.pyplot as plt
from simulater2BodyHigh import SpaceSim2Body

MAX_DV= 0.01166964393
NUM_THRUST=1
filename = "blackBoxHigh.bin"

class earthToMars(base):
    def __init__(self, dim=1*4,c_dim=1):
        super(earthToMars,self).__init__(dim, 0, 1, c_dim, 0, 1e-5)
        # self.set_bounds([0]+[-THRUST]*3 + [30]+[-THRUST]*3+[60]+[-THRUST]*3 ,[120]+[THRUST]*3 +[480]+[THRUST]*3+[480]+[THRUST]*3)
        self.set_bounds(([0]+[-MAX_DV]*3)*1 , ([480]+[MAX_DV]*3)*1)
        self.c_dim=c_dim
        self.dim=dim

    def _objfun_impl(self, x):
        sim=SpaceSim()
        decisions=self._convertToList(x)
        f = sim.simulate(decisions)

        return (-f.particles[3].m,)

    def _compute_constraints_impl(self, x):
        sim=SpaceSim()
        decisions=self._convertToList(x)
        sim2 = sim.simulate(decisions)

        rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
        marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
        distance = ((rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2 +(rocketCord[2]-marsCord[2])**2)**0.5

        ceq = list([0]*self.c_dim)
        ceq[0]=distance-8e-6 # distance-0.01<=0

        return ceq  # distance-0.01<=0

    def _convertToList(self,x):
        dimention =self.dimension/4
        decisions=[]
        for i in range(dimention):
            de=[]
            de.append(x[i*4])
            de.append(x[i*4+1])
            de.append(x[i*4+2])
            de.append(x[i*4+3])
            decisions.append(de)
        return decisions

NUM_THRUST2=2
class earthToMars1(base):
    def __init__(self, dim=NUM_THRUST2*4,c_dim=5):
        super(earthToMars1,self).__init__(dim, 0, 1, c_dim, 0, 1e-5)
        # self.set_bounds([0]+[-THRUST]*3 + [30]+[-THRUST]*3+[60]+[-THRUST]*3 ,[120]+[THRUST]*3 +[480]+[THRUST]*3+[480]+[THRUST]*3)
        self.set_bounds(([0]+[-MAX_DV]*3)*NUM_THRUST2 , ([480]+[MAX_DV]*3)*NUM_THRUST2)
        self.c_dim=c_dim
        self.dim=dim

    def _objfun_impl(self, x):
        sim=SpaceSim()
        decisions=self._convertToList(x)
        f = sim.simulate(decisions)
        return (-f.particles[3].m,)

    def _compute_constraints_impl(self, x):
        sim=SpaceSim()
        decisions=self._convertToList(x)
        sim2 = sim.simulate(decisions)

        distance = self.getDistance(sim2)
        dvx,dvy,dvz = self.getRelativeVelocity(sim2)

        ceq = list([0]*self.c_dim)
        ceq[0]=distance-0.00385032218 # distance-0.01<=0
        ceq[1]=(decisions[0][0]+150)-decisions[0][1]
        ceq[2]=dvx-0.
        ceq[3]=dvy-0.
        ceq[4]=dvz-0.

        return ceq

    def getDistance(self, sim2):
        rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
        marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
        distance = ((rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2 +(rocketCord[2]-marsCord[2])**2)**0.5

        return distance

    def getRelativeVelocity(self, sim2):
        rocketCord = (sim2.particles[3].vx,sim2.particles[3].vy,sim2.particles[3].vz)
        marsCord = (sim2.particles[2].vx,sim2.particles[2].vy,sim2.particles[2].vz)
        return rocketCord[0]-marsCord[0],rocketCord[1]-marsCord[1],rocketCord[2]-marsCord[2]

    def _convertToList(self,x):
        dimention =self.dimension/4
        decisions=[]
        for i in range(dimention):
            de=[]
            de.append(x[i*4])
            de.append(x[i*4+1])
            de.append(x[i*4+2])
            de.append(x[i*4+3])
            decisions.append(de)
        return decisions

MAX_DV2= 0.01166964393
class rocketMars(base):
    def __init__(self, dim=1*4,c_dim=1):
    #USAGE: super(derived_class_name,self).__init__(dim, i_dim, n_obj, c_dim, c_ineq_dim, c_tol)
    #dim: Total dimension of the decision vector
    #i_dim: dimension of the integer part of decision vector (the integer part is placed at the end of the decision vector). Defaults to 0
    #n_obj: number of objectives. Defaults to 1
    #c_dim: total dimension of the constraint vector. dDefaults to 0
    #c_ineq_dim: dimension of the inequality part of the constraint vector (inequality const. are placed at the end of the decision vector). Defaults to 0
    #c_tol: constraint tolerance. When comparing individuals, this tolerance is used to decide whether a constraint is considered satisfied.
        super(rocketMars,self).__init__(dim, 0, 1, c_dim, 0, 1e-8)
        # self.set_bounds([0]+[-THRUST]*3 + [30]+[-THRUST]*3+[60]+[-THRUST]*3 ,[120]+[THRUST]*3 +[480]+[THRUST]*3+[480]+[THRUST]*3)
        self.set_bounds(([480-4]+[-MAX_DV2]*3) , ([480+4]+[MAX_DV2]*3))
        self.c_dim=c_dim
        self.dim=dim

    def _objfun_impl(self, x):
        sim2 = rebound.Simulation.from_file(filename)
        sim=SpaceSim()
        decisions=self._convertToList(x)
        sim2 = sim.simulate1(decisions,sim2)
        dvx,dvy,dvz = self.getRelativeVelocity(sim2)
        dv = (dvx**2+dvy**2+dvz**2)**0.5

        return (dv,)

    def _compute_constraints_impl(self, x):
        sim2 = rebound.Simulation.from_file(filename)
        sim=SpaceSim()
        decisions = self._convertToList(x)
        sim2 = sim.simulate1(decisions,sim2)

        distance = self.getDistance(sim2)
        dvx,dvy,dvz = self.getRelativeVelocity(sim2)
        dv = (dvx**2+dvy**2+dvz**2)**0.5

        ceq = list([0]*self.c_dim)
        ceq[0]=distance-8e-3

        return ceq  # distance-0.01<=0

    def getDistance(self, sim2):
        rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
        marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
        distance = ((rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2 +(rocketCord[2]-marsCord[2])**2)**0.5

        return distance

    def getRelativeVelocity(self, sim2):
        rocketCord = (sim2.particles[3].vx,sim2.particles[3].vy,sim2.particles[3].vz)
        marsCord = (sim2.particles[2].vx,sim2.particles[2].vy,sim2.particles[2].vz)
        return rocketCord[0]-marsCord[0],rocketCord[1]-marsCord[1],rocketCord[2]-marsCord[2]

    def _convertToList(self,x):
        dimention =self.dimension/4
        decisions=[]
        for i in range(dimention):
            de=[]
            de.append(x[i*4])
            de.append(x[i*4+1])
            de.append(x[i*4+2])
            de.append(x[i*4+3])
            decisions.append(de)
        return decisions

MAX_DV3= 0.01166964393
class rocketMars1(base):
    def __init__(self, dim=1*4,c_dim=2):
    #USAGE: super(derived_class_name,self).__init__(dim, i_dim, n_obj, c_dim, c_ineq_dim, c_tol)
    #dim: Total dimension of the decision vector
    #i_dim: dimension of the integer part of decision vector (the integer part is placed at the end of the decision vector). Defaults to 0
    #n_obj: number of objectives. Defaults to 1
    #c_dim: total dimension of the constraint vector. dDefaults to 0
    #c_ineq_dim: dimension of the inequality part of the constraint vector (inequality const. are placed at the end of the decision vector). Defaults to 0
    #c_tol: constraint tolerance. When comparing individuals, this tolerance is used to decide whether a constraint is considered satisfied.
        super(rocketMars1,self).__init__(dim, 0, 1, c_dim, 0, 1e-8)
        # self.set_bounds([0]+[-THRUST]*3 + [30]+[-THRUST]*3+[60]+[-THRUST]*3 ,[120]+[THRUST]*3 +[480]+[THRUST]*3+[480]+[THRUST]*3)
        self.set_bounds(([480-3]+[-MAX_DV3]*3) , ([480+3]+[MAX_DV3]*3))
        self.c_dim=c_dim
        self.dim=dim

    def _objfun_impl(self, x):
        sim2 = rebound.Simulation.from_file(filename)
        sim=SpaceSim()
        decisions=self._convertToList(x)
        f = sim.simulate1(decisions,sim2)
        return (-f.particles[3].m,)

    def _compute_constraints_impl(self, x):
        sim2 = rebound.Simulation.from_file(filename)
        sim=SpaceSim()
        decisions = self._convertToList(x)
        sim2 = sim.simulate1(decisions,sim2)

        distance = self.getDistance(sim2)
        dvx,dvy,dvz = self.getRelativeVelocity(sim2)

        dv = (dvx**2+dvy**2+dvz**2)**0.5

        ceq = list([0]*self.c_dim)

        ceq[0]=distance-8e-3
        ceq[1]=dv-1e-5

        return ceq  # distance-0.01<=0

    def getDistance(self, sim2):
        rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
        marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
        distance = ((rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2 +(rocketCord[2]-marsCord[2])**2)**0.5

        return distance

    def getRelativeVelocity(self, sim2):
        rocketCord = (sim2.particles[3].vx,sim2.particles[3].vy,sim2.particles[3].vz)
        marsCord = (sim2.particles[2].vx,sim2.particles[2].vy,sim2.particles[2].vz)
        return rocketCord[0]-marsCord[0],rocketCord[1]-marsCord[1],rocketCord[2]-marsCord[2]

    def _convertToList(self,x):
        dimention =self.dimension/4
        decisions=[]
        for i in range(dimention):
            de=[]
            de.append(x[i*4])
            de.append(x[i*4+1])
            de.append(x[i*4+2])
            de.append(x[i*4+3])
            decisions.append(de)
        return decisions

def run_example1(n_restarts=5):
    from PyGMO import algorithm, island
    prob = earthToMars()
    algo = algorithm.scipy_slsqp(max_iter=500, acc=1e-5)
    # algo = algorithm.ipopt(max_iter=500, constr_viol_tol=1e-5,dual_inf_tol=1e-5,compl_inf_tol=1e-5)
    # algo.screen_output = True
    algo2 = algorithm.mbh(algo, n_restarts, 0.05)
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
    prob = rocketMars()
    algo = algorithm.scipy_slsqp(max_iter=500, acc=1e-5)
    # algo = algorithm.ipopt(max_iter=500, constr_viol_tol=1e-5,dual_inf_tol=1e-5,compl_inf_tol=1e-5)
    # algo.screen_output = True
    algo2 = algorithm.mbh(algo, n_restarts, 0.05)
    algo2.screen_output = True
    isl = island(algo2, prob, 1)
    print("Running Monotonic Basin Hopping .... this will take a while.")
    isl.evolve(1)
    isl.join()
    print("Is the solution found a feasible trajectory? " +
          str(prob.feasibility_x(isl.population.champion.x)))
    return isl.population.champion.x

def run_example5(n_restarts=5):
    from PyGMO import algorithm, island
    prob = rocketMars()
    algo = algorithm.scipy_slsqp(max_iter=500, acc=1e-5)
    # algo = algorithm.ipopt(max_iter=500, constr_viol_tol=1e-5,dual_inf_tol=1e-5,compl_inf_tol=1e-5)
    # algo.screen_output = True
    algo2 = algorithm.mbh(algo, n_restarts, 0.05)
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
    prob = earthToMars1()
    algo = algorithm.scipy_slsqp(max_iter=500, acc=1e-5)
    # algo = algorithm.ipopt(max_iter=500, constr_viol_tol=1e-5,dual_inf_tol=1e-5,compl_inf_tol=1e-5)
    # algo.screen_output = True
    algo2 = algorithm.mbh(algo, n_restarts, 0.05)
    algo2.screen_output = True
    isl = island(algo2, prob, 1)
    print("Running Monotonic Basin Hopping .... this will take a while.")
    isl.evolve(1)
    isl.join()
    print("Is the solution found a feasible trajectory? " +
          str(prob.feasibility_x(isl.population.champion.x)))
    return isl.population.champion.x

def run_example4(sim,n_restarts=5):
    from PyGMO import algorithm, island
    prob = rocketMars1()
    algo = algorithm.scipy_slsqp(max_iter=500, acc=1e-5)
    # algo = algorithm.ipopt(max_iter=500, constr_viol_tol=1e-5,dual_inf_tol=1e-5,compl_inf_tol=1e-5)
    # algo.screen_output = True
    algo2 = algorithm.mbh(algo, n_restarts, 0.05)
    algo2.screen_output = True
    isl = island(algo2, prob, 1)
    print("Running Monotonic Basin Hopping .... this will take a while.")
    isl.evolve(1)
    isl.join()
    print("Is the solution found a feasible trajectory? " +
          str(prob.feasibility_x(isl.population.champion.x)))
    return isl.population.champion.x

x=earthToMars()
vector=run_example5()
deci=x._convertToList(vector)
print deci
# sim2=SpaceSim()
# sim=sim2.simulate(deci)
# sim.save(filename)
#
# sim2=SpaceSim()
# sim=sim2.animate(deci)

# sim = rebound.Simulation.from_file(filename)
# sim2 = SpaceSim(sim)
# # vector=run_example2(sim)
# vector=run_example2(sim)
# deci=x._convertToList(vector)
# print deci
# deci=[[5.0,0.,0.,0.]]

# sim3 = SpaceSim2Body(sim)
# sim3.animate(deci)


# sim2=SpaceSim2Body(sim)
# sim3=sim2.simulate(sim,deci)
# fig = rebound.OrbitPlot(sim3[0])
# plt.show()

# fig = rebound.OrbitPlot(sim)
# plt.show()


# vector=run_example3()
# deci=x._convertToList(vector)
# print deci
# sim2=SpaceSim()
# sim=sim2.simulate(deci)
# sim.save("blackBoxHigh.bin")
#
# sim2=SpaceSim()
# sim=sim2.animate(deci)
