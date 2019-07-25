import numpy as np

M0 = 450400
m10 = 45040
Re = 6371000
GMe = 3.986004418e14
Htarget = 400000
r2 = Re + Htarget
g0 = 9.80665
gIsp = g0 * 455
h =1.48736155e+05
m = 73672.97286
r1 = h + Re
Dv1 = np.sqrt(GMe / r1) * (np.sqrt((2 * r2) / (r1 + r2)) - 1)
Dv2 = np.sqrt(GMe / r2) * (1 - np.sqrt((2 * r1) / (r1 + r2)))
mf = m/ np.exp((Dv1 + Dv2) / gIsp)

inc = (mf-m10)/M0*100

print(mf, -mf/M0, inc)