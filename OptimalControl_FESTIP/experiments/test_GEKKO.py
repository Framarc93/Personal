import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO

# create GEKKO model
mg = GEKKO()

class Rocket:
    GMe = 3.986004418 * 10**14  # Earth gravitational constant [m^3/s^2]
    Re = 6371.0 * 1000  # Earth Radius [m]
    g0 = 9.80665  # Gravitational acceleration on Earth surface [m/s^2]

    def __init__(self):
        self.Vr = mg.sqrt(self.GMe / self.Re)  # m/s
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
        self.Vtarget = mg.sqrt(self.GMe / self.Rtarget)  # m/s



def air_density(h):
    beta = mg.Param(value=1/8500.0)  # scale factor [1/m]
    rho0 = mg.Param(value=1.225)  # kg/m3
    return rho0*mg.exp(-beta*h)

obj = Rocket()

# scale 0-1 time with tf
mg.time = np.linspace(0,1,401)

# options
mg.options.NODES = 6
mg.options.SOLVER = 3
mg.options.IMODE = 6
mg.options.MAX_ITER = 500
mg.options.MV_TYPE = 0
mg.options.DIAGLEVEL = 0
mg.options.SCALING = 2


# final time
tf = mg.FV(value=1.0,lb=0.1,ub=400)
tf.STATUS = 1

# force
Tr = mg.MV(value=0.0,lb=0.0,ub=obj.Tmax)
Tr.STATUS = 1
Tr.DCOST = 1e-5

Tt = mg.MV(value = 0.0, lb=0.0, ub=obj.Tmax)
Tt.STATUS = 1
Tt.DCOST = 1e-5

# variables
R = mg.Var(value=obj.Re, ub=obj.Re)

teta = mg.Var(value=0.0)
Vr = mg.Var(value=0.0,lb=0,ub=1.7)
Vt = mg.Var(value=0.0,lb=0,ub=1.7)
m = mg.Var(value=obj.M0, ub=obj.M0, lb=(obj.M0-obj.Mp))
rho = mg.CV()
mg.Equation(rho == air_density(R - obj.Re))
Dr = 0.5 * rho * Vr * mg.sqrt(Vr**2 + Vt**2) * obj.Cd * obj.A  # [N]
Dt = 0.5 * rho * Vt * mg.sqrt(Vr**2 + Vt**2) * obj.Cd * obj.A  # [N]
g = obj.g0 * (obj.Re / R)**2  # [m/s2]
g0 = obj.g0
Isp = obj.Isp

# other constraints
q = 0.5 * rho * (Vr**2 + Vt**2)  # [Pa]
# accelaration
a_r = ((Tr - Dr) / m)
a_t = ((Tt - Dt) / m)
#a_mag = mg.FV(value=mg.sqrt(a_r**2 + a_t**2), ub=(obj.MaxG*obj.g0))  # [m/s2]
#a_mag.STATUS = 1
# Thrust
#T = mg.FV(value=mg.sqrt(Tr**2 + Tt**2), ub=obj.Tmax)
#T.STATUS = 1



# differential equations scaled by tf
mg.Equation(R.dt()==Vr)
mg.Equation(teta.dt()==Vt/R)
mg.Equation(Vr.dt()==Tr / m - Dr / m - g + Vt**2 / R)
mg.Equation(Vt.dt()==Tt / m - Dt / m - (Vr * Vt) / R)
mg.Equation(m.dt()==-mg.sqrt(Tr**2 + Tt**2) / g0 / Isp)

# specify endpoint conditions
mg.fix(R, pos=len(mg.time)-1,val=obj.Rtarget)
mg.fix(Vr, pos=len(mg.time)-1,val=0.0)
mg.fix(Vt, pos=len(mg.time)-1,val=obj.Vtarget)

# minimize final time
mg.Obj(tf)

# Optimize launch
mg.solve()

print('Optimal Solution (final time): ' + str(tf.value[0]))

# scaled time
ts = mg.time * tf.value[0]

# plot results
plt.figure(1)
plt.subplot(4,1,1)
plt.plot(ts,R.value,'r-',linewidth=2)
plt.ylabel('Position')
plt.legend(['s (Position)'])

plt.subplot(4,1,2)
plt.plot(ts,Vr.value,'b-',linewidth=2)
plt.plot(ts,Vt.value,'b-',linewidth=2)
plt.ylabel('Velocity')
plt.legend(['v (Velocity)'])

plt.subplot(4,1,3)
plt.plot(ts,m.value,'k-',linewidth=2)
plt.ylabel('Mass')
plt.legend(['m (Mass)'])

plt.subplot(4,1,4)
plt.plot(ts,Tr.value,'g-',linewidth=2)
plt.plot(ts,Tt.value,'g-',linewidth=2)
plt.ylabel('Force')
plt.legend(['u (Force)'])

plt.xlabel('Time')
plt.show()
