from PyGMO.problem import base
import rebound
import numpy as np
import math
from simulatorHigh import SpaceSim

MAX_DV= 0.01166964393
filename = "blackBoxHigh2D.bin"

class earthToMars2D(base):
    def __init__(self, dim=1*3,c_dim=1):
        super(earthToMars2D,self).__init__(dim, 0, 1, c_dim, 0, 1e-5)
        self.set_bounds( ([0]+[-MAX_DV]*2)*1 , ([480]+[MAX_DV]*2)*1)
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
        distance = ((rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2
        +(rocketCord[2]-marsCord[2])**2)**0.5

        ceq = list([0]*self.c_dim)
        # print decisions
        ceq[0]=distance-8e-6 # distance-0.01<=0
        return ceq  # distance-0.01<=0

    def _convertToList(self,x):
        dimention =self.dimension/3
        decisions=[]
        for i in range(dimention):
            de=[]
            de.append(x[i*3])
            de.append(x[i*3+1])
            de.append(x[i*3+2])
            de.append(0.0)
            decisions.append(de)
        return decisions

class earthToMars2DWH(base):
    def __init__(self, dim=1*3,c_dim=1):
        super(earthToMars2DWH,self).__init__(dim, 0, 1, c_dim, 0, 1e-5)
        self.set_bounds( ([0]+[-MAX_DV]*2)*1 , ([480]+[MAX_DV]*2)*1)
        self.c_dim=c_dim
        self.dim=dim

    def _objfun_impl(self, x):
        sim=SpaceSim(wh=True)
        decisions=self._convertToList(x)
        f = sim.simulate(decisions)
        return (-f.particles[3].m,)

    def _compute_constraints_impl(self, x):
        sim=SpaceSim(wh=True)
        decisions=self._convertToList(x)
        sim2 = sim.simulate(decisions)

        rocketCord = (sim2.particles[3].x,sim2.particles[3].y,sim2.particles[3].z)
        marsCord = (sim2.particles[2].x,sim2.particles[2].y,sim2.particles[2].z)
        distance = ((rocketCord[0]-marsCord[0])**2 +(rocketCord[1]-marsCord[1])**2
        +(rocketCord[2]-marsCord[2])**2)**0.5

        ceq = list([0]*self.c_dim)
        # print decisions
        ceq[0]=distance-8e-6 # distance-0.01<=0
        return ceq  # distance-0.01<=0

    def _convertToList(self,x):
        dimention =self.dimension/3
        decisions=[]
        for i in range(dimention):
            de=[]
            de.append(x[i*3])
            de.append(x[i*3+1])
            de.append(x[i*3+2])
            de.append(0.0)
            decisions.append(de)
        return decisions

MAX_DV2= 0.01166964393
class rocketMars2D(base):
    def __init__(self, dim=1*3,c_dim=2):
    #USAGE: super(derived_class_name,self).__init__(dim, i_dim, n_obj, c_dim, c_ineq_dim, c_tol)
    #dim: Total dimension of the decision vector
    #i_dim: dimension of the integer part of decision vector (the integer part is placed at the end of the decision vector). Defaults to 0
    #n_obj: number of objectives. Defaults to 1
    #c_dim: total dimension of the constraint vector. dDefaults to 0
    #c_ineq_dim: dimension of the inequality part of the constraint vector (inequality const. are placed at the end of the decision vector). Defaults to 0
    #c_tol: constraint tolerance. When comparing individuals, this tolerance is used to decide whether a constraint is considered satisfied.
        super(rocketMars2D,self).__init__(dim, 0, 1, c_dim, 0, 1e-8)
        # self.set_bounds([0]+[-THRUST]*3 + [30]+[-THRUST]*3+[60]+[-THRUST]*3 ,[120]+[THRUST]*3 +[480]+[THRUST]*3+[480]+[THRUST]*3)
        self.set_bounds(([480-20]+[-MAX_DV2]*2) , ([480+20]+[MAX_DV2]*2))
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

        ceq = list([0]*self.c_dim)
        ceq[0]=distance-8e-6
        ceq[1]=(dvx**2+dvy**2+dvz**2)**0.5-0.0
        return ceq  # distance-0.01<=0

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

    def _convertToList(self,x):
        dimention =self.dimension/3
        decisions=[]
        for i in range(dimention):
            de=[]
            de.append(x[i*3])
            de.append(x[i*3+1])
            de.append(x[i*3+2])
            de.append(0.0)
            decisions.append(de)
        return decisions

filenameWH = "blackBoxHigh2DWH.bin"
class rocketMars2DWH(base):
    def __init__(self, dim=1*3,c_dim=1):
    #USAGE: super(derived_class_name,self).__init__(dim, i_dim, n_obj, c_dim, c_ineq_dim, c_tol)
    #dim: Total dimension of the decision vector
    #i_dim: dimension of the integer part of decision vector (the integer part is placed at the end of the decision vector). Defaults to 0
    #n_obj: number of objectives. Defaults to 1
    #c_dim: total dimension of the constraint vector. dDefaults to 0
    #c_ineq_dim: dimension of the inequality part of the constraint vector (inequality const. are placed at the end of the decision vector). Defaults to 0
    #c_tol: constraint tolerance. When comparing individuals, this tolerance is used to decide whether a constraint is considered satisfied.
        super(rocketMars2DWH,self).__init__(dim, 0, 1, c_dim, 0, 1e-8)
        # self.set_bounds([0]+[-THRUST]*3 + [30]+[-THRUST]*3+[60]+[-THRUST]*3 ,[120]+[THRUST]*3 +[480]+[THRUST]*3+[480]+[THRUST]*3)
        self.set_bounds(([480-20]+[-MAX_DV2]*2) , ([480+20]+[MAX_DV2]*2))
        self.c_dim=c_dim
        self.dim=dim

    def _objfun_impl(self, x):
        sim2 = rebound.Simulation.from_file(filenameWH)
        sim=SpaceSim(wh=True)
        decisions=self._convertToList(x)
        f = sim.simulate1(decisions,sim2)
        dvx,dvy,dvz = self.getRelativeVelocity(f)
        return ((dvx**2+dvy**2+dvz**2)**0.5,)

    def _compute_constraints_impl(self, x):
        sim2 = rebound.Simulation.from_file(filenameWH)
        sim=SpaceSim(wh=True)
        decisions = self._convertToList(x)
        sim2 = sim.simulate1(decisions,sim2)

        distance = self.getDistance(sim2)
        dvx,dvy,dvz = self.getRelativeVelocity(sim2)

        ceq = list([0]*self.c_dim)
        ceq[0]=distance-8e-6
        # ceq[1]=(dvx**2+dvy**2+dvz**2)**0.5-0.0
        return ceq  # distance-0.01<=0

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

    def _convertToList(self,x):
        dimention =self.dimension/3
        decisions=[]
        for i in range(dimention):
            de=[]
            de.append(x[i*3])
            de.append(x[i*3+1])
            de.append(x[i*3+2])
            de.append(0.0)
            decisions.append(de)
        return decisions

class rocketMars2D2(base):
    def __init__(self, dim=1*3,c_dim=1):
    #USAGE: super(derived_class_name,self).__init__(dim, i_dim, n_obj, c_dim, c_ineq_dim, c_tol)
    #dim: Total dimension of the decision vector
    #i_dim: dimension of the integer part of decision vector (the integer part is placed at the end of the decision vector). Defaults to 0
    #n_obj: number of objectives. Defaults to 1
    #c_dim: total dimension of the constraint vector. dDefaults to 0
    #c_ineq_dim: dimension of the inequality part of the constraint vector (inequality const. are placed at the end of the decision vector). Defaults to 0
    #c_tol: constraint tolerance. When comparing individuals, this tolerance is used to decide whether a constraint is considered satisfied.
        super(rocketMars2D2,self).__init__(dim, 0, 1, c_dim, 0, 1e-8)
        # self.set_bounds([0]+[-THRUST]*3 + [30]+[-THRUST]*3+[60]+[-THRUST]*3 ,[120]+[THRUST]*3 +[480]+[THRUST]*3+[480]+[THRUST]*3)
        self.set_bounds(([480-15]+[-MAX_DV2]*2) , ([480+15]+[MAX_DV2]*2))
        self.c_dim=c_dim
        self.dim=dim

    def _objfun_impl(self, x):
        sim2 = rebound.Simulation.from_file(filename)
        sim=SpaceSim()
        decisions=self._convertToList(x)
        f = sim.simulate1(decisions,sim2)
        dvx,dvy,dvz = self.getRelativeVelocity(f)
        dv = (dvx**2+dvy**2+dvz**2)**0.5
        return (dv,)

    def _compute_constraints_impl(self, x):
        sim2 = rebound.Simulation.from_file(filename)
        sim=SpaceSim()
        decisions = self._convertToList(x)
        sim2 = sim.simulate1(decisions,sim2)

        distance = self.getDistance(sim2)

        ceq = list([0]*self.c_dim)
        ceq[0]=distance-8e-6
        # ceq[1]=(dvx**2+dvy**2+dvz**2)**0.5-0.0

        return ceq  # distance-0.01<=0

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

    def _convertToList(self,x):
        dimention =self.dimension/3
        decisions=[]
        for i in range(dimention):
            de=[]
            de.append(x[i*3])
            de.append(x[i*3+1])
            de.append(x[i*3+2])
            de.append(0.0)
            decisions.append(de)
        return decisions

def run_example1(n_restarts=5):
    from PyGMO import algorithm, island
    prob = earthToMars2D()
    algo = algorithm.scipy_slsqp(max_iter=100, acc=1e-5)
    # algo.screen_output = True
    algo2 = algorithm.mbh(algo, n_restarts, 0.05)
    algo2.screen_output = True
    isl = island(algo2, prob, 1)
    print("Running Monotonic Basin Hopping .... this will take a while.")
    isl.evolve(1)
    isl.join()
    print("Is the solution found a feasible trajectory? " +
          str(prob.feasibility_x(isl.population.champion.x)))
    print("Fitness " +
          str(prob.best_f))
    return isl.population.champion.x

def run_example2(sim,n_restarts=5):
    from PyGMO import algorithm, island
    prob = rocketMars2D()
    algo = algorithm.scipy_slsqp(max_iter=100, acc=1e-5)
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

def run_example3(sim,n_restarts=5):
    from PyGMO import algorithm, island
    prob = rocketMars2D2()
    algo = algorithm.scipy_slsqp(max_iter=100, acc=1e-5)
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

def run_example4(n_restarts=5):
    from PyGMO import algorithm, island
    prob = earthToMars2DWH()
    algo = algorithm.scipy_slsqp(max_iter=100, acc=1e-5)
    # algo.screen_output = True
    algo2 = algorithm.mbh(algo, n_restarts, 0.05)
    algo2.screen_output = True
    isl = island(algo2, prob, 1)
    print("Running Monotonic Basin Hopping .... this will take a while.")
    isl.evolve(1)
    isl.join()
    print("Is the solution found a feasible trajectory? " +
          str(prob.feasibility_x(isl.population.champion.x)))
    print("Fitness " +
          str(prob.best_f))
    return isl.population.champion.x

def run_example5(n_restarts=5):
    from PyGMO import algorithm, island
    prob = rocketMars2DWH()
    algo = algorithm.scipy_slsqp(max_iter=100, acc=1e-5)
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

# Deci is [[319.8305897255627, 0.0016687830320550338, 0.0013691821770248276, 0.0]]
# deci = [[420.2372419060117, 0.010494326408284994, 0.0044382506954818565, 0.0]]
# deci = [[411.21323621411335, 0.008748839048462907, 0.0033290148214346503, 0.0]]
# deci =[[395.8283718212657, 0.006450877568564355, 0.002069880891910152, 0.0]]

# [[301.93091863294785, 0.0016076262089444425, 0.0015896115913838728, 0.0]] ****
# [[320.247591820331, 0.0016738013552311084, 0.0013636072977159816, 0.0]]** WH
# x=earthToMars2D()
# # vector=run_example4()
# vector=run_example1()
# deci=x._convertToList(vector)
# print deci
# print vector
# deci=x._convertToList(vector)
# print deci
# sim2=SpaceSim()
# sim=sim2.simulate(deci)
# import rebound
# import matplotlib.pyplot as plt
# fig = rebound.OrbitPlot(sim)
# plt.show()
# sim2=SpaceSim()
# sim=sim2.simulate(deci)
# sim.save(filename)

# vector=run_example5()

# sim = rebound.Simulation.from_file(filename)
# # vector=run_example3(sim)
# deci=x._convertToList(vector)
# print deci
