import numpy as np
import random
import matplotlib.pyplot as plt
from SLSQP_EqNotMapped import *

def mask_MS_2(var):
    try:
        eq_Cond, ineq_Cond, objective = MultiShooting(var, dynamicsInt)

        J = objective + max(ineq_Cond) + sum(abs(eq_Cond))
        if np.isnan(J) or not (np.isreal(J)):
            J = 1e6 + random.randint * 1e6
    except:
        J = 1e8 + random.randint * 1e8

    return J

X0a = np.load("opt.npy")
J = mask_MS_2(X0a)
x28 = 0.60
vvv = np.array((x28, J))
while x28 <= 0.7:
    X0a[28] = x28
    J = mask_MS_2(X0a)
    vvv = np.vstack((vvv, np.array((x28, J))))
    x28 = x28 + 0.000005
plt.figure()
plt.xlabel("alpha [deg]")
plt.ylabel("Objective function")
plt.plot(vvv[:, 0], vvv[:, 1])
plt.show()
