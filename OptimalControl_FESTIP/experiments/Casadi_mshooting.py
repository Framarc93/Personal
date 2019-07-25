from casadi import *
import numpy as np
from scipy import integrate


def air_density(h):
    beta = 1 / 8500.0  # scale factor [1/m]
    rho0 = 1.225  # kg/m3
    return rho0 * np.exp(-beta * h)

N = 50

opti = casadi.Opti()

# ---- decision variables ---------
X = opti.variable(5,N+1)
R   = X[0,:]
theta = X[1,:]
Vr    = X[2,:]
Vt    = X[3,:]
m     = X[4,:]
U = opti.variable(2,N+1)
Tr = U[0,:]
Tt = U[1,:]
time = opti.variable()

# ---- objective          ---------
opti.minimize(m[0]-m[N])

Re = opti.parameter()
opti.set_value(Re, 6371000)

rho = air_density(R - Re)

Cd = opti.parameter()
opti.set_value(Cd, 0.6)
A = opti.parameter()
opti.set_value(A, 4.0)
Dr = 0.5 * rho * Vr * sqrt(Vr**2 + Vt**2) * Cd * A  # [N]
Dt = 0.5 * rho * Vt * sqrt(Vr**2 + Vt**2) * Cd * A  # [N]

g0 = opti.parameter()
opti.set_value(g0, 9.81)

g = g0 * (Re / R)**2  # [m/s2]

Isp = opti.parameter()
opti.set_value(Isp, 300)

q = 0.5 * rho * (Vr**2 + Vt**2)  # [Pa]

# accelaration
a_r = (Tr - Dr) / m
a_t = (Tt - Dt) / m
a_mag = sqrt(a_r**2 + a_t**2)  # [m/s2]

# Thrust
T = sqrt(Tr**2 + Tt**2)

# ---- dynamic constraints --------
f = lambda x,u,i: vertcat(x[2],
                        x[3]/x[0],
                        u[0]/x[4] - Dr[i]/x[4] - g[i] + (x[3]**2)/x[0],
                        u[1]/x[4] - Dt[i]/x[4] - (x[2]*x[3])/x[0],
                        -sqrt(u[0]**2+u[1]**2)/(g0*Isp))


dt = time/N # length of a control interval

for k in range(N): # loop over control intervals
   # Runge-Kutta 4 integration
   #k1 = f(X[:,k],         U[:,k])
   #k2 = f(X[:,k]+dt/2*k1, U[:,k])
   #k3 = f(X[:,k]+dt/2*k2, U[:,k])
   #k4 = f(X[:,k]+dt*k3,   U[:,k])
   x_next = X[:,k] + dt * f(X[:,k], U[:,k], k)
   opti.subject_to(X[:,k+1]==x_next) # close the gaps