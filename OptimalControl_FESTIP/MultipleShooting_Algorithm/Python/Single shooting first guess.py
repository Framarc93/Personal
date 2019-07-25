import numpy as np
import matplotlib.pyplot as plt
from models import *
from scipy.interpolate import splrep, splev, PchipInterpolator
from scipy.integrate import solve_ivp
import sys
import os
sys.path.insert(0, 'home/francesco/git_workspace/FESTIP_Work/Collocation_algorithm')

class Spaceplane:

    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = np.deg2rad(5.2)  # deg latitude
        self.longstart = np.deg2rad(-52.775)  # deg longitude
        self.chistart = np.deg2rad(113)  # deg flight direction
        self.incl = np.deg2rad(51.6)  # deg orbit inclination
        self.gammastart = np.deg2rad(89.9)  # deg
        self.M0 = 450400  # kg  starting mass
        self.g0 = 9.80665  # m/s2
        self.gIsp = self.g0 * 455  # g0 * Isp max
        self.omega = 7.2921159e-5
        self.MaxQ = 40000  # Pa
        self.MaxAx = 30  # m/s2
        self.MaxAz = 15  # m/s2
        self.Htarget = 400000  # m target height after hohmann transfer
        self.wingSurf = 500.0  # m2
        self.lRef = 34.0  # m
        self.k = 5e3  # [Nm] livello di precisione per trimmaggio
        self.m10 = self.M0 * 0.1
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.Hini = 190000
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(self.incl) / np.cos(self.latstart))
        self.varOld = np.zeros((0))
        self.costOld = np.zeros((0))
        self.eqOld = np.zeros((0))
        self.ineqOld = np.zeros((0))
        self.States = np.zeros((0))
        self.Controls = np.zeros((0))

Nstates = 7
obj = Spaceplane()

'''reading of aerodynamic coefficients and specific impulse from file'''

cl = fileReadOr("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/clfile.txt")
cd = fileReadOr("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cdfile.txt")
cm = fileReadOr("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cmfile.txt")
cl = np.asarray(cl)
cd = np.asarray(cd)
cm = np.asarray(cm)

with open("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/impulse.dat") as f:
    impulse = []
    for line in f:
        line = line.split()
        if line:
            line = [float(i) for i in line]
            impulse.append(line)

f.close()

presv = []
spimpv = []

for i in range(len(impulse)):
    presv.append(impulse[i][0])
    spimpv.append(impulse[i][1])

presv = np.asarray(presv)
spimpv = np.asarray(spimpv)

v = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/v.npy")
chi = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/chi.npy")
gamma = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/gamma.npy")
teta = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/teta.npy")
lam = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/lambda.npy")
h = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/h.npy")
m = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/m.npy")
alfa_guess=np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/alfa.npy")

delta_guess=np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/delta.npy")
deltaf_guess=np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/deltaf.npy")
#deltaf_guess = np.deg2rad(deltaf_guess)
tau_guess=np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/tau.npy")
mu_guess=np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/mu.npy")
#mu_guess = np.deg2rad(mu_guess)
time_vector=np.load("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/timeCol.npy")
#control = fileRead("LASTopt5maxf2.out")
#control = np.asarray((control))

Nguess = len(alfa_guess)
#at = np.transpose(control[:,2])
#dt = np.transpose(control[:,1])
#dft = np.transpose(control[:,4])
#tt = np.transpose(control[:,3])
controls = np.vstack((alfa_guess, delta_guess, deltaf_guess, tau_guess, mu_guess))

states_init = np.array((v[0], chi[0], gamma[0], teta[0], lam[0], h[0], m[0]))


def dynamicsInt(t, states, alfa_int, delta_int, deltaf_int, tau_int, mu_int):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''

    v = states[0]
    chi = states[1]
    gamma = states[2]
    #teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = alfa_int(t)
    delta = delta_int(t)
    deltaf = deltaf_int(t)
    tau = tau_int(t)
    mu = mu_int(t)

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)
    # Press = np.asarray(Press, dtype=np.float64)
    # rho = np.asarray(rho, dtype=np.float64)
    # c = np.asarray(c, dtype=np.float64)

    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                            obj.xcg0, obj.xcgf, obj.pref)
    # L = np.asarray(L, dtype=np.float64)
    # D = np.asarray(D, dtype=np.float64)
    # MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
                                obj.lRef, obj.xcgf, obj.xcg0)

    # T = np.asarray(T, dtype=np.float64)
    # isp = np.asarray(isp, dtype=np.float64)
    # Deps = np.asarray(Deps, dtype=np.float64)
    # MomT = np.asarray(MomT, dtype=np.float64)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = obj.g0 * (obj.Re / (obj.Re + h)) ** 2
    # g = np.asarray(g, dtype=np.float64)

    dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                   (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   ((T * np.sin(eps) + L) * np.sin(mu)) / (m * v * np.cos(gamma)) - np.cos(gamma) * np.cos(chi) *
                   np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
                           np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                   - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi),
                   ((T * np.sin(eps) + L) * np.cos(mu)) / (m * v) - (g / v - v / (obj.Re + h)) * np.cos(
                       gamma) + 2 * obj.omega \
                   * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                   (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                   -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                   np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                   v * np.sin(gamma),
                   -T / (g0 * isp)))
    dx = np.reshape(dx, (7,))
    return dx


def SingleShooting(states, controls, dyn, time, Nint):
    '''this function integrates the dynamics equation over time. It takes as input the vector of variables and the dynamics equation set'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''

    #Time = time

    x = np.zeros((Nint, Nstates))

    x[:Nstates] = states[0:Nstates]  # vector of intial states ready

    # now interpolation of controls

    alfa_Int = PchipInterpolator(time, controls[0, :])
    delta_Int = PchipInterpolator(time, controls[1, :])
    deltaf_Int = PchipInterpolator(time, controls[2, :])
    tau_Int = PchipInterpolator(time, controls[3, :])
    mu_Int = PchipInterpolator(time, controls[4, :])

    time_new = np.linspace(0, time[-1], Nint)

    dt = (time_new[1] - time_new[0])

    t = time_new

    #sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int), t_span=[0, time[-1]], y0=x,
     #               t_eval=time_new, method='RK45')

    for i in range(Nint - 1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt * dyn(t[i], x[i, :], alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k1: ", k1)
        k2 = dt * dyn(t[i] + dt / 2, x[i, :] + k1 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k2: ", k2)
        k3 = dt * dyn(t[i] + dt / 2, x[i, :] + k2 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k3: ", k3)
        k4 = dt * dyn(t[i + 1], x[i, :] + k3, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k4: ", k4)
        x[i + 1, :] = x[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


    #vres = sol.y[0, :]
    #chires = sol.y[1, :]
    #gammares = sol.y[2, :]
    #tetares = sol.y[3, :]
    #lamres = sol.y[4, :]
    #hres = sol.y[5, :]
    #mres = sol.y[6, :]
    vres = x[:, 0]
    chires = x[:, 1]
    gammares = x[:, 2]
    tetares = x[:, 3]
    lamres = x[:, 4]
    hres = x[:, 5]
    mres = x[:, 6]
    alfares = alfa_Int(time_new)
    deltares = delta_Int(time_new)
    deltafres = deltaf_Int(time_new)
    taures = tau_Int(time_new)
    mures = mu_Int(time_new)

    return vres, chires, gammares, tetares, lamres, hres, mres, time_new, alfares, deltares, deltafres, taures, mures


Nint = 10000
tstart = 0
tfin = time_vector[-1]

vres, chires, gammares, tetares, lamres, hres, mres, time, alfares, deltares, deltafres, taures, mures = SingleShooting(states_init, controls, dynamicsInt, time_vector, Nint)


rep = len(time)
Press, rho, c = isaMulti(hres, obj.psl, obj.g0, obj.Re)
Press = np.asarray(Press, dtype=np.float64)
rho = np.asarray(rho, dtype=np.float64)
c = np.asarray(c, dtype=np.float64)
M = vres / c

L, D, MomA = aeroForcesMulti(M, alfares, deltafres, cd, cl, cm, vres, obj.wingSurf, rho, obj.lRef, obj.M0, mres, obj.m10, obj.xcg0, obj.xcgf, obj.pref, rep)
L = np.asarray(L, dtype=np.float64)
D = np.asarray(D, dtype=np.float64)
MomA = np.asarray(MomA, dtype=np.float64)

T, Deps, isp, MomT = thrustMulti(Press, mres, presv, spimpv, deltares, taures, rep, obj.psl, obj.M0, obj.m10, obj.lRef, obj.xcgf, obj.xcg0)
T = np.asarray(T, dtype=np.float64)
Deps = np.asarray(Deps, dtype=np.float64)
MomT = np.asarray(MomT, dtype=np.float64)

MomTot = MomA + MomT

g0 = obj.g0
eps = Deps + alfares

# dynamic pressure

q = 0.5 * rho * (vres ** 2)

# accelerations

ax = (T * np.cos(Deps) - D * np.cos(alfares) + L * np.sin(alfares)) / mres
az = (T * np.sin(Deps) + D * np.sin(alfares) + L * np.cos(alfares)) / mres

r1 = hres + obj.Re
Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
mf = mres / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / obj.gIsp))

downrange = (vres**2)/g0 * np.sin(2 * gammares)


'''time_stat = np.linspace(0, time_vector[-1], 20)
v_int = splrep(time, vres, s=0)
v_new = splev(time_stat, v_int, der=0)
chi_int = splrep(time, chires, s=0)
chi_new = splev(time_stat, chi_int, der=0)
gamma_int = splrep(time, gammares, s=0)
gamma_new = splev(time_stat, gamma_int, der=0)
teta_int = splrep(time, tetares, s=0)
teta_new = splev(time_stat, teta_int, der=0)
lam_int = splrep(time, lamres, s=0)
lam_new = splev(time_stat, lam_int, der=0)
h_int = splrep(time, hres, s=0)
h_new = splev(time_stat, h_int, der=0)
m_int = splrep(time, mres, s=0)
m_new = splev(time_stat, m_int, der=0)
alfa_int = splrep(time, np.rad2deg(alfares), s=301)
alfa_new = splev(time_stat, alfa_int, der=0)
delta_int = splrep(time, deltares, s=0)
delta_new = splev(time_stat, delta_int, der=0)
deltaf_int = splrep(time, np.rad2deg(deltafres), s=200)
deltaf_new = splev(time_stat, deltaf_int, der=0)
tau_int = splrep(time, taures, s=0)
tau_new = splev(time_stat, tau_int, der=0)
mu_int = splrep(time, np.rad2deg(mures), s=0)
mu_new = splev(time_stat, mu_int, der=0)
np.save("v", v_new)
np.save("chi", chi_new)
np.save("gamma", gamma_new)
np.save("teta", teta_new)
np.save("lambda", lam_new)
np.save("h", h_new)
np.save("m", m_new)
np.save("alfa", alfa_new)
np.save("delta", delta_new)
np.save("deltaf", deltaf_new)
np.save("tau", tau_new)
np.save("mu", mu_new)
np.save("time", time_stat)
plt.figure()
plt.plot(time_stat, v_new)
plt.figure()
plt.plot(time_stat, chi_new)
plt.figure()
plt.plot(time_stat, gamma_new)
plt.figure()
plt.plot(time_stat, teta_new)
plt.figure()
plt.plot(time_stat, lam_new)
plt.figure()
plt.plot(time_stat, h_new)
plt.figure()
plt.plot(time_stat, m_new)
plt.figure()
plt.plot(time_stat, alfa_new)
plt.figure()
plt.plot(time_stat, delta_new)
plt.figure()
plt.plot(time_stat, deltaf_new)
plt.figure()
plt.plot(time_stat, tau_new)
plt.figure()
plt.plot(time_stat, mu_new)

v_interpOG = interpolate.PchipInterpolator(time, vres)
chi_interpOG = interpolate.PchipInterpolator(time, chires)
gamma_interpOG = interpolate.PchipInterpolator(time, gammares)
teta_interpOG = interpolate.PchipInterpolator(time, tetares)
lam_interpOG = interpolate.PchipInterpolator(time, lamres)
h_interpOG = interpolate.PchipInterpolator(time, hres)
m_interpOG = interpolate.PchipInterpolator(time, mres)
v = v_interpOG(time_vector)
chi = chi_interpOG(time_vector)
gamma = gamma_interpOG(time_vector)
teta = teta_interpOG(time_vector)
lam = lam_interpOG(time_vector)
h = h_interpOG(time_vector)
m = m_interpOG(time_vector)
np.save("v", vres)
np.save("chi", chires)
np.save("gamma", gammares)
np.save("teta", tetares)
np.save("lam", lamres)
np.save("h", hres)
np.save("m", mres)
np.save("time", time)



'''
# ------------------------
# Visualization
plt.figure()
plt.title("Altitude profile")
plt.plot(time, hres/1000, label="Altitude")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
plt.legend(loc="best")


plt.figure()
plt.title("Velocity")
plt.plot(time, vres, label="V")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend(loc="best")


plt.figure()
plt.title("Mass")
plt.plot(time, mres, label="Mass")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Mass [kg]")
plt.legend(loc="best")


plt.figure()
plt.title("Acceleration")
plt.plot(time, ax, label="Acc x")
plt.plot(time, az, label="Acc z")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Acceleration [m/s2]")
plt.legend(loc="best")


plt.figure()
plt.title("Throttle profile")
plt.step(time, deltares, where='post', label="Delta")
plt.step(time, taures, where='post', label="Tau")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" % ")
plt.legend(loc="best")


plt.figure()
plt.title("Angle of attack profile")
plt.step(time, np.rad2deg(alfares), where='post', label="Alfa")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")


plt.figure()
plt.title("Body Flap deflection profile")
plt.step(time, np.rad2deg(deltafres), where='post', label="Delta f")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")


plt.figure()
plt.title("Trajectory angles")
plt.plot(time, np.rad2deg(chires), label="Chi")
plt.plot(time, np.rad2deg(gammares), label="Gamma")
plt.plot(time, np.rad2deg(tetares), label="Teta")
plt.plot(time, np.rad2deg(lamres), label="Lambda")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")


plt.figure()
plt.title("Chi")
plt.plot(time, np.rad2deg(chires), label="Chi")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")


plt.figure()
plt.title("Gamma")
plt.plot(time, np.rad2deg(gammares), label="Gamma")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")


plt.figure()
plt.title("Teta")
plt.plot(time, np.rad2deg(tetares), label="Teta")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")


plt.figure()
plt.title("Lambda")
plt.plot(time, np.rad2deg(lamres), label="Lambda")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")


plt.figure()
plt.title("Dynamic pressure profile")
plt.plot(time, q/1000, label="Q")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" kPa ")
plt.legend(loc="best")


plt.figure()
plt.title("Moment")
plt.plot(time, MomTot / 1000, label="Total Moment")
plt.axhline(5, 0, time[-1], color='r')
plt.axhline(-5, 0, time[-1], color='r')
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" kNm ")
plt.legend(loc="best")


plt.figure()
plt.title("Trajectory")
plt.plot(downrange/1000, hres/1000, label="Trajectory")
plt.grid()
plt.xlabel("km")
plt.ylabel(" km ")
plt.legend(loc="best")

plt.figure()
plt.title("Mach profile")
plt.plot(time, M, label="Mach")
plt.grid()
plt.xlabel("time [s]")
plt.legend(loc="best")


plt.figure()
plt.title("Lift, Drag and Thrust profile")
plt.plot(time, L, label="Lift")
plt.plot(time, D, label="Drag")
plt.plot(time, T, label="Thrust")
plt.grid()
plt.xlabel("time [s]")
plt.legend(loc="best")


plt.show()

