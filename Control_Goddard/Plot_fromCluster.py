from scipy.integrate import solve_ivp
import numpy as np
import operator
import pygraphviz as pgv
import random
from deap import gp
import matplotlib.pyplot as plt
import sys
import timeit
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import multiprocessing
from scipy.interpolate import PchipInterpolator
import matplotlib.animation as animation
from matplotlib import style
import datetime
import time

time_tot = np.load("tottime.npy")
tformat = str(datetime.timedelta(seconds=int(time_tot)))
print("This simulation took: ", tformat)

class Rocket:
    GMe = 3.986004418 * 10**14  # Earth gravitational constant [m^3/s^2]
    Re = 6371.0 * 1000  # Earth Radius [m]
    g0 = 9.80665  # Gravitational acceleration on Earth surface [m/s^2]

    def __init__(self):
        self.M0 = 5000  # Initial total mass [kg]
        self.Mc = 0.4  # Initial Propellant mass over total mass
        self.Cd = 0.2  # Drag Coefficient [-]
        self.area = 10  # area [m2]
        self.Isp = 300.0  # Isp [s]
        self.max_thrust = 2  # maximum thrust to initial weight ratio
        self.Tmax = self.max_thrust * self.M0 * self.g0

    def air_density(self, h):
        beta = 1/8500.0  # scale factor [1/m]
        rho0 = 1.225  # kg/m3
        return rho0*np.exp(-beta*h)

obj = Rocket()
Nstates = 3
Ncontrols = 1

Rref = np.load("R.npy")
Vref = np.load("V.npy")
mref = np.load("m.npy")
tref = np.load("time.npy")
tfin = tref[-1]

Rfun = PchipInterpolator(tref, Rref)
Vfun = PchipInterpolator(tref, Vref)
mfun = PchipInterpolator(tref, mref)

del Rref, Vref, mref, tref

rout = np.load("rout.npy")
vout = np.load("vout.npy")
mout = np.load("mout_initial")
ttgp = np.load("time_initial")
Tplot1 = np.load("thrust_initial")

rR = Rfun(ttgp)
vR = Vfun(ttgp)
mR = mfun(ttgp)

rout_c = np.load("rout_middle")
vout_c = np.load("vout_middle")
mout_c = np.load("mout_middle")
ttgp_c = np.load("time_middle")
Tplot2 = np.load("thrust_middle")
rout_gp = np.load("rout_final")
vout_gp = np.load("vout_final")
mout_gp = np.load("mout_final")
ttgp_gp = np.load("time_final")
Tplot3 = np.save("thrust_final")


plt.figure(1)
plt.plot(ttgp, (rR - obj.Re) / 1e3, 'r--', label="SET POINT")
plt.axhline(0, 0, ttgp[-1], color='r')
plt.plot(ttgp, (rout - obj.Re) / 1e3, marker='.', color = 'k', label="ON DESIGN")
plt.plot(ttgp_c, (rout_c - obj.Re) / 1e3, marker='.', color = 'b', label="EVALUATION TIME")
plt.plot(ttgp_gp, (rout_gp - obj.Re) / 1e3, marker='.', color = 'g', label="ON DESIGN 2")
plt.figure(2)
plt.plot(ttgp, vR, 'r--', label="SET POINT")
plt.plot(ttgp, vout, marker='.', color = 'k', label="ON DESIGN")
plt.plot(ttgp_c, vout_c, marker='.', color = 'b', label="EVALUATION TIME")
plt.plot(ttgp_gp, vout_gp, marker='.', color = 'g', label="ON DESIGN 2")
plt.figure(3)
plt.plot(ttgp, mR, 'r--', label="SET POINT")
plt.axhline(obj.M0*obj.Mc, 0, ttgp[-1], color='r')
plt.plot(ttgp, mout, marker='.', color = 'k', label="ON DESIGN")
plt.plot(ttgp_c, mout_c, marker='.', color = 'b', label="EVALUATION TIME")
plt.plot(ttgp_gp, mout_gp, marker='.', color = 'g', label="ON DESIGN 2")
plt.figure(4)
plt.axhline(obj.Tmax, 0, ttgp[-1], color='r')
plt.axhline(0, 0, ttgp[-1], color='r')
plt.plot(ttgp, Tplot1, marker='.', color = 'k', label="ON DESIGN")
plt.plot(ttgp_c, Tplot2, marker='.', color = 'b', label="EVALUATION TIME")
plt.plot(ttgp_gp, Tplot3, marker='.', color = 'g', label="ON DESIGN 2")

plt.show(block=True)