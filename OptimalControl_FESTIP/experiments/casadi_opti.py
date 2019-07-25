# Car race along a track
# ----------------------
# An optimal control problem (OCP),
# solved with direct multiple-shooting.
#
# For more information see: http://labs.casadi.org/OCP
from casadi import *
import numpy as np
from scipy import integrate

class Rocket:
   GMe = 3.986004418 * 10 ** 14  # Earth gravitational constant [m^3/s^2]
   Re = 6371.0 * 1000  # Earth Radius [m]
   g0 = 9.80665  # Gravitational acceleration on Earth surface [m/s^2]

   def __init__(self):
      self.Vr = np.sqrt(self.GMe / self.Re)  # m/s
      self.H0 = 10.0  # m
      self.V0 = 0.0
      self.M0 = 100000.0  # kg
      self.Mp = self.M0 * 0.99
      self.Cd = 0.6
      self.A = 4.0  # m2
      self.Isp = 300.0  # s
      self.g0 = 9.80665  # m/s2
      self.Tmax = self.M0 * self.g0 * 1.5
      self.MaxQ = 14000.0  # Pa
      self.MaxG = 8.0  # G
      self.Htarget = 400.0 * 1000  # m
      self.Rtarget = self.Re + self.Htarget  # m/s
      self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s


def air_density(h):
   beta = 1/8500.0  # scale factor [1/m]
   rho0 = 1.225  # kg/m3
   return rho0*np.exp(-beta*h)


obj = Rocket()

N = 50 # number of control intervals

opti = casadi.Opti() # Optimization problem

# ---- decision variables ---------
X = opti.variable(5,N+1) # state trajectory
R   = X[0,:]
theta = X[1,:]
Vr    = X[2,:]
Vt    = X[3,:]
m     = X[4,:]
U = opti.variable(2,N+1)# control trajectory (throttle)
Tr = U[0,:]
Tt = U[1,:]
time = opti.variable()      # final time

# ---- objective          ---------
opti.minimize(m[0]-m[N])
Re = opti.parameter()
opti.set_value(Re, obj.Re)
# other variables

rho = air_density(R - Re)

Cd = opti.parameter()
opti.set_value(Cd, obj.Cd)
A = opti.parameter()
opti.set_value(A, obj.A)

Dr = 0.5 * rho * Vr * sqrt(Vr**2 + Vt**2) * Cd * A  # [N]
Dt = 0.5 * rho * Vt * sqrt(Vr**2 + Vt**2) * Cd * A  # [N]

g0 = opti.parameter()
opti.set_value(g0, obj.g0)
g = g0 * (Re / R)**2  # [m/s2]

Isp = opti.parameter()
opti.set_value(Isp, obj.Isp)
q = 0.5 * rho * (Vr**2 + Vt**2)  # [Pa]
# accelaration
a_r = (Tr - Dr) / m
a_t = (Tt - Dt) / m
a_mag = sqrt(a_r**2 + a_t**2)  # [m/s2]
# Thrust
T = sqrt(Tr**2 + Tt**2)

# ---- dynamic constraints --------
fl = lambda x,u,i: vertcat(x[2],
                        x[3]/x[0],
                        u[0]/x[4] - Dr[i]/x[4] - g[i] + (x[3]**2)/x[0],
                        u[1]/x[4] - Dt[i]/x[4] - (x[2]*x[3])/x[0],
                        -sqrt(u[0]**2+u[1]**2)/(g0*Isp)) # dx/dt = f(x,u)


dt = time/N # length of a control interval

for k in range(N): # loop over control intervals
   # Runge-Kutta 4 integration
   k1 = fl(X[:,k],         U[:,k], k)
   k2 = fl(X[:,k]+dt/2*k1, U[:,k], k)
   k3 = fl(X[:,k]+dt/2*k2, U[:,k], k)
   k4 = fl(X[:,k]+dt*k3,   U[:,k], k)
   x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4)
   opti.subject_to(X[:,k+1]==x_next) # close the gaps

# ---- normalization --------------

unit_R = 1#Re
unit_theta = 1
unit_V = 1#np.sqrt(obj.GMe / Re)
unit_m = 1#obj.M0
unit_t = 1#unit_R / unit_V
unit_T = 1#unit_m * unit_R / unit_t ** 2

# ---- path constraints -----------

opti.subject_to(R>Re/unit_R)
opti.subject_to(opti.bounded((obj.M0 - obj.Mp)/unit_m, m, obj.M0/unit_m))
opti.subject_to(opti.bounded(0.0, Tr, obj.Tmax/unit_T))
opti.subject_to(opti.bounded(0.0, Tt, obj.Tmax/unit_T))
opti.subject_to(T<=obj.Tmax/unit_T)
opti.subject_to(a_mag<=(obj.MaxG * g0))

# ---- boundary conditions --------
opti.subject_to(R[0]==Re/unit_R)   # start at position 0 ...
opti.subject_to(theta[0]==0/unit_theta)
opti.subject_to(Vr[0]==0.0/unit_V)
opti.subject_to(Vt[0]==0.0/unit_V)
opti.subject_to(m[0]==obj.M0/unit_m)
opti.subject_to(R[-1]==obj.Rtarget/unit_R)
opti.subject_to(Vr[-1]==0.0/unit_V)
opti.subject_to(Vt[-1]==obj.Vtarget/unit_V)
opti.subject_to(time>=0) # Time must be positive

# ---- initial values for solver ---
opti.set_initial(R, 1)
opti.set_initial(theta, 1)
opti.set_initial(Vr, 1)
opti.set_initial(Vt, 1)
opti.set_initial(m, 1)
opti.set_initial(Tr, 1)
opti.set_initial(Tt, 1)
opti.set_initial(time, 1)


# ---- solve NLP              ------
opti.solver("ipopt") # set numerical backend
sol = opti.solve()   # actual solve


# ---- post-processing        ------
from pylab import plot, step, figure, legend, show, spy

figure()
plot(sol.value(R),label="pos")
figure()
plot(sol.value(Vt),label="speed")
plot(sol.value(m),label="mass")
step(range(N),sol.value(Tt),'k',label="throttle")
legend(loc="best")

figure()
spy(sol.value(jacobian(opti.g,opti.x)))
figure()
spy(sol.value(hessian(opti.f+dot(opti.lam_g,opti.g),opti.x)[0]))

show()

