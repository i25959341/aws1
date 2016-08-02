from simulatorHigh import SpaceSim
import rebound
import numpy as np
import math
import matplotlib.pyplot as plt

adaptive=[]

for i in range(10):
    x=SpaceSim()
    deci = [[5.368545621384487e-06, 2.2642820923555172e-05, 0.00010579608330588702, 0.0], [250.00000536854563, 0.0013151829722556129, -0.002883903806489052, 0.0]]

    import time
    t0 = time.time()
    x.simulate(deci)
    t1 = time.time()

    total = t1-t0

    adaptive.append(total)

print np.average(adaptive)
import time

steps = np.linspace(0.1, 20, 20/0.1)
inx=[]
totalTime=[]



for i, step in enumerate(steps):
    dummy=[]
    print i
    print step
    for j in range(5):
        x=SpaceSim()
        deci = [[5.368545621384487e-06, 2.2642820923555172e-05, 0.00010579608330588702, 0.0], [250.00000536854563, 0.0013151829722556129, -0.002883903806489052, 0.0]]

        t0 = time.time()
        x.simulateRefine(deci,step)
        t1 = time.time()

        total = t1-t0
        dummy.append(total)

    inx.append(step)
    totalTime.append(np.average(dummy))


fig = plt.figure()
fig.suptitle('Time/simulation vs Timestep', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)

ax.set_xlabel('Timestep size (Day)')
ax.set_ylabel('Time/simulation average (Sec)')
plt.xticks([x*2 for x in range(12)])
ax.plot(inx, totalTime, "o")

plt.show()
