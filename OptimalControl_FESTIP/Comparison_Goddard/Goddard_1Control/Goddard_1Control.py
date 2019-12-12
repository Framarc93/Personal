# -*- coding: utf-8 -*-
# Copyright 2017 Interstellar Technologies Inc. All Rights Reserved.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Problem, Guess, Condition, Dynamics
from scipy import interpolate, integrate

class Rocket:
    GMe = 3.986004418 * 10**14  # Earth gravitational constant [m^3/s^2]
    Re = 6371.0 * 1000  # Earth Radius [m]
    g0 = 9.80665  # Gravitational acceleration on Earth surface [m/s^2]

    def __init__(self):
        self.M0 = 500000  # Initial total mass [kg]
        self.Mc = 0.4  # Initial Propellant mass over total mass
        self.Cd = 0.2  # Drag Coefficient [-]
        self.area = 2*np.pi*(5.4/2)*52 + np.pi*(5.4/2)**2  # area [m2]
        self.Isp = 400.0  # Isp [s]
        self.max_thrust = 2  # maximum thrust to initial weight ratio

    def air_density(self, h):
        beta = 1/8500.0  # scale factor [1/m]
        rho0 = 1.225  # kg/m3
        return rho0*np.exp(-beta*h)


def dynamics(prob, obj, section):
    R = prob.states(0, section)
    v = prob.states(1, section)
    m = prob.states(2, section)
    T = prob.controls(0, section)

    rho = obj.air_density(R - obj.Re)
    drag = 0.5 * rho * v ** 2 * obj.Cd * obj.area
    g = obj.GMe / R**2
    g0 = obj.g0
    Isp = obj.Isp

    dx = Dynamics(prob, section)
    dx[0] = v
    dx[1] = (T - drag) / m - g
    dx[2] = - T / g0 / Isp
    return dx()


def equality(prob, obj):
    R = prob.states_all_section(0)
    v = prob.states_all_section(1)
    m = prob.states_all_section(2)
    T = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = Condition()

    # event condition
    result.equal(R[0], obj.Re)
    result.equal(v[0], 0.0)
    result.equal(m[0], obj.M0)
    result.equal(v[-1], 0.0)
    result.equal(m[-1], obj.M0 * obj.Mc)

    return result()


def inequality(prob, obj):
    R = prob.states_all_section(0)
    v = prob.states_all_section(1)
    m = prob.states_all_section(2)
    T = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = Condition()
    # lower bounds
    result.lower_bound(R, obj.Re)
    result.lower_bound(v, 0.0)
    result.lower_bound(m, obj.M0 * obj.Mc)
    result.lower_bound(T, 0.0)
    result.lower_bound(tf, 10)
    # upper bounds
    result.upper_bound(m, obj.M0)
    result.upper_bound(T, obj.max_thrust * obj.M0 * obj.g0)

    return result()


def cost(prob, obj):
    R = prob.states_all_section(0)
    return -R[-1] / obj.Re


def cost_derivative(prob, obj):
    jac = Condition(prob.number_of_variables)
    index_R_end = prob.index_states(0, 0, -1)
    jac.change_value(index_R_end, -1)
    return jac()

# ========================
plt.close("all")
plt.ion()
# Program Starting Point
time_init = [0.0, 600]
n = [30]
Nint = 10000
Npoints = n[0]
Nstates = 3
Ncontrols = 1
num_states = [3]
num_controls = [1]
max_iteration = 50

flag_savefig = True
savefig_file = "Single_"

# ------------------------
# set OpenGoddard class for algorithm determination
prob = Problem(time_init, n, num_states, num_controls, max_iteration)

# ------------------------
# create instance of operating object
obj = Rocket()

# ------------------------
# set designer unit
unit_R = obj.Re
unit_v = np.sqrt(obj.GMe / obj.Re)
unit_m = obj.M0
unit_t = unit_R / unit_v
unit_T = unit_m * unit_R / unit_t**2
prob.set_unit_states_all_section(0, unit_R)
prob.set_unit_states_all_section(1, unit_v)
prob.set_unit_states_all_section(2, unit_m)
prob.set_unit_controls_all_section(0, unit_T)
prob.set_unit_time(unit_t)

# ========================
# Initial parameter guess

# altitude profile
R_init = Guess.cubic(prob.time_all_section, obj.Re, 0.0, obj.Re+50*1000, 0.0)
# Guess.plot(prob.time_all_section, R_init, "Altitude", "time", "Altitude")
# if(flag_savefig):plt.savefig(savefig_file + "guess_alt" + ".png")

# velocity
V_init = Guess.linear(prob.time_all_section, 0.0, 0.0)
# Guess.plot(prob.time_all_section, V_init, "Velocity", "time", "Velocity")

# mass profile
M_init = Guess.cubic(prob.time_all_section, obj.M0, -0.6, obj.M0*obj.Mc, 0.0)
# Guess.plot(prob.time_all_section, M_init, "Mass", "time", "Mass")
# if(flag_savefig):plt.savefig(savefig_file + "guess_mass" + ".png")

# thrust profile
T_init = Guess.cubic(prob.time_all_section, obj.max_thrust * obj.M0 * obj.g0, 0.0, 0.0, 0.0)
# Guess.plot(prob.time_all_section, T_init, "Thrust Guess", "time", "Thrust")
# if(flag_savefig):plt.savefig(savefig_file + "guess_mass" + ".png")

plt.show()

# ========================
# Substitution initial value to parameter vector to be optimized
prob.set_states_all_section(0, R_init)
prob.set_states_all_section(1, V_init)
prob.set_states_all_section(2, M_init)
prob.set_controls_all_section(0, T_init)

# ========================
# Main Process
# Assign problem to SQP solver
prob.dynamics = [dynamics]
prob.knot_states_smooth = []
prob.cost = cost
# prob.cost_derivative = cost_derivative
prob.equality = equality
prob.inequality = inequality


def display_func():
    R = prob.states_all_section(0)
    print("max altitude: {0:.5f}".format(R[-1] - obj.Re))

prob.solve(obj, display_func, ftol=1e-12)


def dynamicsInt(t, states, T_int):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''
    R = states[0]
    v = states[1]
    m = states[2]
    T = T_int(t)

    rho = obj.air_density(R - obj.Re)
    drag = 0.5 * rho * v ** 2 * obj.Cd * obj.area
    g = obj.GMe / R ** 2
    g0 = obj.g0
    Isp = obj.Isp

    dx = np.array((v,
                   (T - drag) / m - g,
                   - T / g0 / Isp))

    return dx


def SingleShooting(states, controls, dyn, time, Nint):
    '''this function integrates the dynamics equation over time. It takes as input the vector of variables and the dynamics equation set'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''

    #Time = time

    x = np.zeros((Nint, Nstates))

    x[0,:] = states[0:Nstates]  # vector of intial states ready

    # now interpolation of controls

    T_Int = interpolate.PchipInterpolator(time, controls)

    time_new = np.linspace(0, time[-1], Nint)

    dt = (time_new[1] - time_new[0])

    t = time_new

    #sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int), t_span=[0, time[-1]], y0=x,
     #               t_eval=time_new, method='RK45')

    for i in range(Nint - 1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt * dyn(t[i], x[i, :], T_Int)
        # print("k1: ", k1)
        k2 = dt * dyn(t[i] + dt / 2, x[i, :] + k1 / 2, T_Int)
        # print("k2: ", k2)
        k3 = dt * dyn(t[i] + dt / 2, x[i, :] + k2 / 2, T_Int)
        # print("k3: ", k3)
        k4 = dt * dyn(t[i + 1], x[i, :] + k3, T_Int)
        # print("k4: ", k4)
        x[i + 1, :] = x[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


    Rres = x[:, 0]
    Vres = x[:, 1]
    mres = x[:, 2]
    Tres = T_Int(time_new)

    return Rres, Vres, mres, Tres, time_new


# ========================
# Post Process
# ------------------------
# Convert parameter vector to variable
R = prob.states_all_section(0)
v = prob.states_all_section(1)
m = prob.states_all_section(2)
T = prob.controls_all_section(0)
time = prob.time_update()

Uval = np.array(T)

Xval = np.array((R, v, m))

X = np.zeros((0))
for i in range(Npoints):
    '''creation of vector of states initial guesses'''
    for j in range(Nstates):
        X = np.hstack((X, Xval[j][i]))

Rres, Vres, mres, Tres, tres = SingleShooting(X, Uval, dynamicsInt, time, Nint)

np.save("R", Rres)
np.save("V", Vres)
np.save("m", mres)
np.save("T", Tres)
np.save("time", tres)

# ------------------------
# Calculate necessary variables
rho = obj.air_density(R - obj.Re)
drag = 0.5 * rho * v ** 2 * obj.Cd * obj.area
g = obj.GMe / R**2

# ------------------------
# Visualizetion
plt.figure()
plt.title("Altitude profile")
plt.plot(time, (R - obj.Re)/1000, "o", label="Altitude")
plt.plot(tres, (Rres-obj.Re)/1000, label="Integration")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
if(flag_savefig): plt.savefig(savefig_file + "altitude" + ".png")

plt.figure()
plt.title("Velocity")
plt.plot(time, v, "o", label="Velocity")
plt.plot(tres, Vres, label="V Integration")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Velocity [m/s]")
if(flag_savefig): plt.savefig(savefig_file + "velocity" + ".png")

plt.figure()
plt.title("Mass")
plt.plot(time, m, "o", label="Mass")
plt.plot(tres, mres, label="Integration")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Mass [kg]")
if(flag_savefig): plt.savefig(savefig_file + "mass" + ".png")

plt.figure()
plt.title("Thrust profile")
plt.plot(time, T / 1000, marker="o", label="Thrust")
#plt.plot(time, drag / 1000, marker="o", label="Drag")
#plt.plot(time, m * g / 1000, marker="o", label="Gravity")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Thrust [kN]")
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_file + "force" + ".png")

plt.show()