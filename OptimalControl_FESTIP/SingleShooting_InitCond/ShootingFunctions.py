import numpy as np
from scipy.interpolate import PchipInterpolator
import constraints as Const
import sys
sys.path.insert(0, 'home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP')
import models as mod

def SingleShooting(var, dyn, Nint, Nstates, NContPoints, obj, varStates, Ncontrols, cl, cd, cm, presv, spimpv, NineqCond, cont_init):

    global penalty
    '''this function integrates the dynamics equation over time.
    INPUT: states: states vector
              controls: controls matrix
              dyn: dynamic equations
              tstart: initial time
              tfin: final time
              Nint: unmber of integration steps
    states and controls must be given with real values!!! not scaled!!!
    needed a fixed step integrator
    tstart and tfin are the initial and final time of the considered leg'''

    penalty = False

    varD = var * (obj.UBV - obj.LBV) + obj.LBV  # variable vector back to original dimensions

    alfa = np.zeros((NContPoints))
    delta = np.zeros((NContPoints))
    # deltaf = np.zeros((NContPoints))
    # tau = np.zeros((NContPoints))
    # mu = np.zeros((NContPoints))

    tfin = varD[-1]

    for k in range(NContPoints):
        '''this for loop takes the controls from the optimization variable and stores them into different variables'''
        '''here controls are scaled'''
        alfa[k] = varD[varStates + Ncontrols * k]
        delta[k] = varD[varStates + 1 + Ncontrols * k]
        # deltaf[k] = varD[varStates + i * (Ncontrols * NContPoints) + 2 + Ncontrols * k]
        # tau[k] = varD[varStates + i * (Ncontrols * NContPoints) + 2 + Ncontrols * k]
        # mu[k] = varD[varStates + i * (Ncontrols * NContPoints) + 4 + Ncontrols * k]

    controls = np.vstack((alfa, delta))  # np.vstack((delta))#, deltaf, tau, mu))  # orig intervals

    timeCont = np.linspace(0.0, tfin, NContPoints)
    x = np.zeros((Nint, 7))

    alfa_Int = PchipInterpolator(timeCont, controls[0, :])
    delta_Int = PchipInterpolator(timeCont, controls[1, :])
    # deltaf_Int = interpolate.PchipInterpolator(timeCont, controls[2, :])
    # tau_Int = interpolate.PchipInterpolator(timeCont, controls[2, :])
    # mu_Int = interpolate.PchipInterpolator(timeCont, controls[4, :])

    t = np.linspace(0.0, tfin, Nint)
    dt = (t[1] - t[0])

    x[0, :] = np.array((obj.vstart, varD[0], obj.gammastart, obj.thetastart, obj.lamstart, obj.hstart, obj.M0))

    for c in range(Nint - 1):
        k1 = dt * dyn(t[c], x[c, :], alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        k2 = dt * dyn(t[c] + dt / 2, x[c, :] + k1 / 2, alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        k3 = dt * dyn(t[c] + dt / 2, x[c, :] + k2 / 2, alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        k4 = dt * dyn(t[c + 1], x[c, :] + k3, alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        x[c + 1, :] = x[c, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    t_ineq = np.linspace(0.0, tfin, NineqCond)

    vres = x[:, 0].T  # orig interavals
    chires = x[:, 1].T
    gammares = x[:, 2].T
    tetares = x[:, 3].T
    lamres = x[:, 4].T
    hres = x[:, 5].T
    mres = x[:, 6].T
    '''penalty = False
    for i in range(len(vres)):
        if np.isnan(vres[i]):
            vres[i] = obj.vmin
            penalty = True
        elif np.isinf(vres[i]):
            vres[i] = obj.vmax
            penalty = True
        if np.isnan(chires[i]):
            chires[i] = obj.chimin
            penalty = True
        elif np.isinf(chires[i]):
            chires[i] = obj.chimax
            penalty = True
        if np.isnan(gammares[i]):
            gammares[i] = obj.gammamin
            penalty = True
        elif np.isinf(gammares[i]):
            gammares[i] = obj.gammamax
            penalty = True
        if np.isnan(tetares[i]):
            tetares[i] = obj.tetamin
            penalty = True
        elif np.isinf(tetares[i]):
            tetares[i] = obj.tetamax
            penalty = True
        if np.isnan(lamres[i]):
            lamres[i] = obj.lammin
            penalty = True
        elif np.isinf(lamres[i]):
            lamres[i] = obj.lammax
            penalty = True
        if np.isnan(hres[i]):
            hres[i] = obj.hmin
            penalty = True
        elif np.isinf(hres[i]):
            hres[i] = obj.hmax
            penalty = True'''
    vres = np.nan_to_num(vres)
    v_Int = PchipInterpolator(t, vres)
    v_ineq = v_Int(t_ineq)
    chires = np.nan_to_num(chires)
    chi_Int = PchipInterpolator(t, chires)
    chi_ineq = chi_Int(t_ineq)
    gammares = np.nan_to_num(gammares)
    gamma_Int = PchipInterpolator(t, gammares)
    gamma_ineq = gamma_Int(t_ineq)
    tetares = np.nan_to_num(tetares)
    teta_Int = PchipInterpolator(t, tetares)
    teta_ineq = teta_Int(t_ineq)
    lamres = np.nan_to_num(lamres)
    lam_Int = PchipInterpolator(t, lamres)
    lam_ineq = lam_Int(t_ineq)
    hres = np.nan_to_num(hres)
    h_Int = PchipInterpolator(t, hres)
    h_ineq = h_Int(t_ineq)

    m_Int = PchipInterpolator(t, mres)
    m_ineq = m_Int(t_ineq)
    alfares = alfa_Int(t).T
    alfa_Int_post = PchipInterpolator(t, alfares)
    alfa_ineq = alfa_Int_post(t_ineq)
    deltares = delta_Int(t).T
    delta_Int_post = PchipInterpolator(t, deltares)
    delta_ineq = delta_Int_post(t_ineq)
    #deltafres = np.zeros(len(t))  # deltaf_Int(time_old)
    #taures = np.zeros(len(t))  # tau_Int(time_old)
    #mures = np.zeros(len(t))  # mu_Int(time_old)

    states_after = np.column_stack((vres, chires, gammares, tetares, lamres, hres, mres))
    controls_after = np.column_stack((alfares, deltares))  # , deltafrescol, taurescol, murescol))
    states_ineq = np.column_stack((v_ineq, chi_ineq, gamma_ineq, teta_ineq, lam_ineq, h_ineq, m_ineq))
    controls_ineq = np.column_stack((alfa_ineq, delta_ineq))  # , deltafrescol, taurescol, murescol))
    obj.States = states_after
    obj.Controls = controls_after

    ineq_c = Const.inequalityAll(states_ineq, controls_ineq, NineqCond, obj, cl, cd, cm, presv, spimpv)

    eq_c = Const.equality(varD, varStates, cont_init, states_after[-1,:], controls_after[-1,:], obj, cl, cd, cm, presv, spimpv)

    h = states_after[-1, 5]
    m = states_after[-1, 6]
    delta = controls_after[-1, 1]
    tau = 0.0  # controls_after[-1, 2]

    Press, rho, c = mod.isa(h, obj.psl, obj.g0, obj.Re)

    T, Deps, isp, MomT = mod.thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10, obj.lRef,
                                    obj.xcgf, obj.xcg0)

    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / np.exp((Dv1 + Dv2) / (obj.g0 * isp))

    '''pen = []
    for i in range(len(vres)):
        #if vres[i] > obj.vmax:
        #    pen.append((vres[i]-obj.vmax)/obj.vmax)
        #elif vres[i] < obj.vmin:
        #    pen.append((vres[i] - obj.vmin) / obj.vmax)
        #if chires[i] > obj.chimax:
        #    pen.append((chires[i]-obj.chimax)/obj.chimax)
        #elif chires[i] < obj.chimin:
        #    pen.append((chires[i] - obj.chimin) / obj.chimax)
        #if gammares[i] > obj.gammamax:
        #    pen.append((gammares[i]-obj.gammamax)/obj.gammamax)
        #elif gammares[i] < obj.gammamin:
        #    pen.append((gammares[i] - obj.gammamin) / obj.gammamax)
        #if hres[i] > obj.hmax:
        #    pen.append((hres[i]-obj.hmax)/obj.hmax)
        if hres[i] < obj.hmin:
            pen.append((hres[i] - obj.hmin) / obj.hmax)'''
    cost = -mf / obj.M0  # + sum(abs(np.array(pen)))

    '''if penalty:
        cost = cost/100
        eq_c = eq_c*100
        ineq_c = ineq_c*100'''

    obj.costOld = cost
    obj.eqOld = eq_c
    obj.varOld = var
    obj.ineqOld = ineq_c

    return ineq_c, eq_c, cost


