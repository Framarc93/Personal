import numpy as np
from scipy.interpolate import PchipInterpolator
from functools import partial
import constraintsMS as cons
from models import *
import multiprocessing

def SingleShootingMulti(i, var, dyn, Nint, NContPoints, Nstates, varTot, Ncontrols, varStates, obj, cl, cd, cm, presv, spimpv, states_init):
    '''this function integrates the dynamics equation over time.'''
    '''INPUT: states: states vector
              controls: controls matrix
              dyn: dynamic equations
              tstart: initial time
              tfin: final time
              Nint: unmber of integration steps'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''
    '''tstart and tfin are the initial and final time of the considered leg'''
    var = np.nan_to_num(var)
    alfa = np.zeros((NContPoints))
    delta = np.zeros((NContPoints))
    #deltaf = np.zeros((NContPoints))
    #tau = np.zeros((NContPoints))
    #mu = np.zeros((NContPoints))
    if i == 0:
        states = states_init
        states[1] = var[0]
    else:
        states = var[1 + (i-1) * Nstates:i * Nstates + 1]  # orig intervals

    tstart = 0
    if i != 0:
        for j in range(i):
            tstart = tstart + var[varTot + j]
    tfin = tstart + var[varTot + i]

    for k in range(NContPoints):
        '''this for loop takes the controls from the optimization variable and stores them into different variables'''
        '''here controls are scaled'''
        alfa[k] = var[varStates + i * (Ncontrols * NContPoints) + Ncontrols * k]
        delta[k] = var[varStates + i * (Ncontrols * NContPoints) + 1 + Ncontrols * k]
        #deltaf[k] = var[varStates + i * (Ncontrols * NContPoints) + 2 + Ncontrols * k]
        #tau[k] = var[varStates + i * (Ncontrols * NContPoints) + 2 + Ncontrols * k]
        #mu[k] = var[varStates + i * (Ncontrols * NContPoints) + 4 + Ncontrols * k]

    controls = np.vstack((alfa, delta))#, deltaf, tau, mu))  # orig intervals

    #print("single shooting")
    tstart = np.nan_to_num(tstart)
    tfin = np.nan_to_num(tfin)
    if tfin <= tstart:
        tfin = tfin + 1
    timeCont = np.linspace(tstart, tfin, NContPoints)
    x = np.zeros((Nint, Nstates))
    # vector of intial states ready
    # now interpolation of controls

    try:
        alfa_Int = PchipInterpolator(timeCont, np.nan_to_num(controls[0, :]))
    except ValueError:
        print("E")
    delta_Int = PchipInterpolator(timeCont, np.nan_to_num(controls[1, :]))
    #deltaf_Int = PchipInterpolator(timeCont, controls[2, :])
    #tau_Int = PchipInterpolator(timeCont, controls[2, :])
    #mu_Int = PchipInterpolator(timeCont, controls[4, :])

    time_int = np.linspace(tstart, tfin, Nint)
    dt = (time_int[1] - time_int[0])
    t = time_int
    x[0,:] = states

    for c in range(Nint-1):
        k1 = dt*dyn(t[c], x[c, :], obj, cl, cd, cm, presv, spimpv, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        k2 = dt*dyn(t[c] + dt / 2, x[c, :] + k1 / 2, obj, cl, cd, cm, presv, spimpv, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        k3 = dt*dyn(t[c] + dt / 2, x[c, :] + k2 / 2, obj, cl, cd, cm, presv, spimpv, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        k4 = dt*dyn(t[c + 1], x[c, :] + k3, obj, cl, cd, cm, presv, spimpv, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        x[c + 1, :] = x[c, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    vres = x[:, 0]  # orig interavals
    chires = x[:, 1]
    gammares = x[:, 2]
    tetares = x[:, 3]
    lamres = x[:, 4]
    hres = x[:, 5]
    mres = x[:, 6]
    alfares = alfa_Int(time_int)
    deltares = delta_Int(time_int)
    deltafres = np.zeros(len(time_int)) #deltaf_Int(time_old)
    taures = np.zeros(len(time_int)) #tau_Int(time_old)
    mures = np.zeros(len(time_int)) #mu_Int(time_old)

    return vres, chires, gammares, tetares, lamres, hres, mres, time_int, alfares, deltares, deltafres, taures, mures



def SingleShooting(states, controls, dyn, tstart, tfin, Nint, NContPoints, Nstates, obj, cl, cd, cm, presv, spimpv):
    '''this function integrates the dynamics equation over time.'''
    '''INPUT: states: states vector
              controls: controls matrix
              dyn: dynamic equations
              tstart: initial time
              tfin: final time
              Nint: unmber of integration steps'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''
    '''tstart and tfin are the initial and final time of the considered leg'''


    timeCont = np.linspace(tstart, tfin, NContPoints)
    x = np.zeros((Nint, Nstates))

    # vector of intial states ready
    # now interpolation of controls

    alfa_Int = PchipInterpolator(timeCont, controls[0, :])
    delta_Int = PchipInterpolator(timeCont, controls[1, :])
    #deltaf_Int = interpolate.PchipInterpolator(timeCont, controls[2, :])
    #tau_Int = interpolate.PchipInterpolator(timeCont, controls[2, :])
    #mu_Int = interpolate.PchipInterpolator(timeCont, controls[4, :])


    time_int = np.linspace(tstart, tfin, Nint)
    dt = (time_int[1] - time_int[0])
    t = time_int
    x[0,:] = states

    for c in range(Nint-1):
        k1 = dt*dyn(t[c], x[c, :], obj, cl, cd, cm, presv, spimpv, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        k2 = dt*dyn(t[c] + dt / 2, x[c, :] + k1 / 2, obj, cl, cd, cm, presv, spimpv, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        k3 = dt*dyn(t[c] + dt / 2, x[c, :] + k2 / 2, obj, cl, cd, cm, presv, spimpv, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        k4 = dt*dyn(t[c + 1], x[c, :] + k3, obj, cl, cd, cm, presv, spimpv, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        x[c + 1, :] = x[c, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    vres = x[:, 0]  # orig interavals
    chires = x[:, 1]
    gammares = x[:, 2]
    tetares = x[:, 3]
    lamres = x[:, 4]
    hres = x[:, 5]
    mres = x[:, 6]
    alfares = alfa_Int(time_int)
    deltares = delta_Int(time_int)
    deltafres = np.zeros(len(time_int)) #deltaf_Int(time_old)
    taures = np.zeros(len(time_int)) #tau_Int(time_old)
    mures = np.zeros(len(time_int)) #mu_Int(time_old)

    return vres, chires, gammares, tetares, lamres, hres, mres, time_int, alfares, deltares, deltafres, taures, mures#, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int
