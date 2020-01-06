from scipy.interpolate import PchipInterpolator
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def smooth_init(leg, contp):
    v = np.load("v.npy")
    v = v[-1]
    chi= np.load("chi.npy")
    chi = chi[-1]
    gamma = np.load("gamma.npy")
    gamma = gamma[-1]
    teta = np.load("teta.npy")
    teta = teta[-1]
    lam = np.load("lam.npy")
    lam = lam[-1]
    h = np.load("h.npy")
    h = h[-1]
    m = np.load("m.npy")
    m = m[-1]
    alfa = np.load("alfa.npy")
    alfa = alfa[-1]
    delta = np.load("delta.npy")
    delta = delta[-1]
    #deltaf = np.load("deltaf.npy")
    #deltaf = deltaf[-1]
    tau = np.load("tau.npy")
    tau = tau[-1]
    #mu = np.load("mu.npy")
    #mu = mu[-1]
    time = np.load("time.npy")
    time = time[-1]

    v = list(v)
    chi = list(chi)
    gamma = list(gamma)
    teta = list(teta)
    lam = list(lam)
    h = list(h)
    m = list(m)
    alfa = list(alfa)
    delta = list(delta)
    tau = list(tau)
    time = list(time)

    to_remove = list(np.where(np.diff(time) == 0)[0])
    while len(to_remove) != 0:
        for i in to_remove:
            v.pop(i)
            chi.pop(i)
            gamma.pop(i)
            teta.pop(i)
            lam.pop(i)
            h.pop(i)
            m.pop(i)
            alfa.pop(i)
            delta.pop(i)
            tau.pop(i)
            time.pop(i)
        to_remove = list(np.where(np.diff(time) == 0)[0])

    tf = time[-1]

    time_cont = np.linspace(0, tf, leg*contp)
    time_stat = np.linspace(0, tf, leg+1)

    v_int = PchipInterpolator(time, v)
    v_new = v_int(time_stat)
    chi_int = PchipInterpolator(time, chi)
    chi_new = chi_int(time_stat)
    gamma_int = PchipInterpolator(time, gamma)
    gamma_new = gamma_int(time_stat)
    teta_int = PchipInterpolator(time, teta)
    teta_new = teta_int(time_stat)
    lam_int = PchipInterpolator(time, lam)
    lam_new = lam_int(time_stat)
    h_int = PchipInterpolator(time, h)
    h_new = h_int(time_stat)
    m_int = PchipInterpolator(time, m)
    m_new = m_int(time_stat)
    alfa_int = PchipInterpolator(time, alfa)
    alfa_new = alfa_int(time_cont)
    delta_int = PchipInterpolator(time, delta)
    delta_new = delta_int(time_cont)
    #deltaf_int = splrep(time, deltaf, s=1)
    #deltaf_new = splev(time_cont, deltaf_int, der=0)
    tau_int = PchipInterpolator(time, tau)
    tau_new = tau_int(time_cont)
    #mu_int = splrep(time, mu, s=1)
    #mu_new = splev(time_cont, mu_int, der=0)

    '''plt.figure()
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
    plt.plot(time_cont, alfa_new)
    plt.figure()
    plt.plot(time_cont, delta_new)
    plt.figure()
    plt.plot(time_cont, deltaf_new)
    plt.figure()
    plt.plot(time_cont, tau_new)
    plt.figure()
    plt.plot(time_cont, mu_new)
    plt.show()'''

    return v_new, chi_new, gamma_new, teta_new, lam_new, h_new, m_new, alfa_new, delta_new, tau_new #, deltaf_new, tau_new, mu_new

def bound_def(X, U, uplimx, inflimx, uplimu, inflimu):
    lbs = np.zeros((len(X)))
    ubs = np.zeros((len(X)))
    lbc = np.zeros((len(U)))
    ubc = np.zeros((len(U)))
    for i in range(len(X)):
        if X[i] == 0:
            lbs[i] = inflimx[i] * 10 / 100
            ubs[i] = uplimx[i] * 10 / 100
        else:
            lbs[i] = X[i] * (1 - 50 / 100)
            ubs[i] = X[i] * (1 + 50 / 100)
        if lbs[i] < inflimx[i]:
            lbs[i] = inflimx[i]
        if ubs[i] > uplimx[i]:
            ubs[i] = uplimx[i]
    for j in range(len(U)):
        if U[j] == 0:
            lbc[j] = inflimu[j] * 10 / 100
            ubc[j] = uplimu[j] * 10 / 100
        else:
            lbc[j] = U[j] * (1 - 50 / 100)
            ubc[j] = U[j] * (1 + 50 / 100)
        if lbc[j] < inflimu[j]:
            lbc[j] = inflimu[j]
        if ubc[j] > uplimu[j]:
            ubc[j] = uplimu[j]

    return lbs, lbc, ubs, ubc
