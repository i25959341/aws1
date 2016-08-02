from PyGMO.problem import base
import rebound
import numpy as np
import math
from simulator import SpaceSim
from blackBox import earthToMars
from blackBox2D import earthToMars2D

def run(n_restarts=5):
    from PyGMO import algorithm, island, population
    prob = earthToMars()
    algo = algorithm.scipy_slsqp(max_iter=500, acc=1e-5)
    # algo.screen_output = True
    algo2 = algorithm.mbh(algo, n_restarts, 0.005)
    algo2.screen_output = True

    pop = population(prob)

    pop.push_back((-3.810404036178629e-05, 8.036667366434322e-05, 0, -0.00011631957922519811, -0.0003960700040113729, 0, 0.00014343900668268246, -0.00039460589468829016, 0, 0.0004133243825183847, -0.0002676479632615287, 0, -6.353773946676955e-05, -0.0004027302771161609, 0, -0.00019483461157664088, -0.0003938299142410649, 0, 0.0003740376551173652, -0.00045439735580127933, 0, 0.00026271994456226056, -4.17413726080276e-05, 0, 0.0004025294016016401, 9.22186764465555e-05, 0, 0.0004379362102351141, -8.202101747983173e-05, 0, 2.0842990495214604e-05, -1.927554372930426e-05, 0, -2.392388475139966e-05, -6.3420840462436174e-06, 0))
    pop.push_back((0.00018354551497353738, 0.0002897005581203533, 0, 9.385407683672441e-05, -0.0004375546286935724, 0, -0.00017406053466786356, -0.0004055793819144533, 0, 7.811816626063441e-05, -0.00028869842254392053, 0, 0.000280132941671916, -0.00045467528344872834, 0, 0.00031161406626870487, -0.0004418005074233615, 0, 0.00016912620000403375, -0.00045156036938030775, 0, 0.00043500734938167605, -4.4611940286304056e-05, 0, 0.00023373694896547512, 4.622353180355802e-06, 0, 0.00043504614537196785, -0.00042017445674379463, 0, 0.00016822207354911628, 0.00010574669088542543, 0, 2.1129649656070842e-05, 0.00020199652091584146, 0))
    # pop.push_back((301.93091863294785, 0.0016076262089444425, 0.0015896115913838728, 0.0))
    # pop.push_back((420.2372419060117, 0.010494326408284994, 0.0044382506954818565, 0.0))
    # pop.push_back((411.21323621411335, 0.008748839048462907, 0.0033290148214346503, 0.0))
    # pop.push_back((395.8283718212657, 0.006450877568564355, 0.002069880891910152, 0.0))
    # pop.push_back((319.95400029222867, 0.0016702166037494744, 0.0013676901851197968, 0.0))
    # pop.push_back((319.5113399461457, 0.00166499548529299, 0.0013736935829129556, 0.0))
    # pop.push_back((320.0969905134936, 0.001671977113629641, 0.001365741362825864, 0.0))
    # pop.push_back((324.8947207784664, 0.0017420256877963634, 0.0013024051696600683, 0.0))

    isl = island(algo2,pop)
    print("Running Monotonic Basin Hopping .... this will take a while.")
    isl.evolve(1)
    isl.join()
    print("Is the solution found a feasible trajectory? " +
          str(prob.feasibility_x(isl.population.champion.x)))
    return isl.population.champion.x

vector = run()
