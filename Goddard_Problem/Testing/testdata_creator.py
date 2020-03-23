import matplotlib.pyplot as plt
import numpy as np

cases = []
for t in np.linspace(20, 200, 50):
    '''if t<80:
        for cd in np.arange(0.61, 1, 0.085):
            cases.append([cd, t])
    else:'''
    for cd in np.linspace(0.61, 2.0, 20):
        cases.append([cd, t])
cases = np.array(cases)
plt.figure(0)
plt.plot(cases[:, 1], cases[:, 0], '.')
plt.show(block=True)
np.save("testdata_cd.npy", cases)
