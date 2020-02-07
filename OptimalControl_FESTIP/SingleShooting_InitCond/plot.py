import numpy as np
import matplotlib.pyplot as plt
import models as mod
from scipy.interpolate import PchipInterpolator
from dynamics import dynamicsInt
import os

def plot(var, Nint, NContPoints, obj, Nstates, varStates, Ncontrols, cl, cm ,cd, presv, spimpv, flag_save, savedata_file, savefig_file, maxiter, timestr, tformat):

    #time = np.zeros((1))
    #timeTotal = np.zeros((0))
    alfaCP = np.zeros((NContPoints))
    deltaCP = np.zeros((NContPoints))
    #deltafCP = np.zeros((Nleg, NContPoints))
    #tauCP = np.zeros((Nleg, NContPoints))
    #muCP = np.zeros((Nleg, NContPoints))

    timestart = 0.0
    varD = var * (obj.UBV - obj.LBV) + obj.LBV

    alfa = np.zeros((NContPoints))
    delta = np.zeros((NContPoints))
    #deltaf = np.zeros((NContPoints))
    #tau = np.zeros((NContPoints))
    #mu = np.zeros((NContPoints))

    tfin = varD[-1]
    timeTotal = np.linspace(timestart, tfin, Nint)
    tC = np.linspace(timestart, tfin, NContPoints)

    for k in range(NContPoints):
        alfa[k] = varD[varStates + Ncontrols * k]
        alfaCP[k] = alfa[k]
        delta[k] = varD[varStates + 1 + Ncontrols * k]
        deltaCP[k] = delta[k]
        #deltaf[k] = varD[varStates + i * (Ncontrols * NContPoints) + 2 + Ncontrols * k]
        #deltafCP[i, k] = deltaf[k]
        #tau[k] = varD[varStates + i * (Ncontrols * NContPoints) + 2 + Ncontrols * k]
        #tauCP[i, k] = tau[k]
        #mu[k] = varD[varStates + i * (Ncontrols * NContPoints) + 4 + Ncontrols * k]
        #muCP[i, k] = mu[k]
    controls = np.vstack((alfa, delta))#, deltaf, tau, mu)) # orig intervals


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
        k1 = dt * dynamicsInt(t[c], x[c, :], alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        k2 = dt * dynamicsInt(t[c] + dt / 2, x[c, :] + k1 / 2, alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        k3 = dt * dynamicsInt(t[c] + dt / 2, x[c, :] + k2 / 2, alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        k4 = dt * dynamicsInt(t[c + 1], x[c, :] + k3, alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        x[c + 1, :] = x[c, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    vres = x[:, 0].T
    chires = x[:, 1].T
    gammares = x[:, 2].T
    tetares = x[:, 3].T
    lamres = x[:, 4].T
    hres = x[:, 5].T
    mres = x[:, 6].T
    alfares = alfa_Int(t).T
    deltares = delta_Int(t).T
    deltafres = np.zeros((len(vres)))
    taures = np.zeros((len(vres)))
    rep = len(vres)
    '''for i in range(len(vres)):
        if np.isnan(vres[i]):
            vres[i] = obj.vmin
        elif np.isinf(vres[i]):
            vres[i] = obj.vmax
        if np.isnan(chires[i]):
            chires[i] = obj.chimin
        elif np.isinf(chires[i]):
            chires[i] = obj.chimax
        if np.isnan(gammares[i]):
            gammares[i] = obj.gammamin
        elif np.isinf(gammares[i]):
            gammares[i] = obj.gammamax
        if np.isnan(tetares[i]):
            tetares[i] = obj.tetamin
        elif np.isinf(tetares[i]):
            tetares[i] = obj.tetamax
        if np.isnan(lamres[i]):
            lamres[i] = obj.lammin
        elif np.isinf(lamres[i]):
            lamres[i] = obj.lammax
        if np.isnan(hres[i]):
            hres[i] = obj.hmin
        elif np.isinf(hres[i]):
            hres[i] = obj.hmax'''

    Press, rho, c = mod.isaMulti(hres, obj.psl, obj.g0, obj.Re)
    Press = np.asarray(Press, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    M = vres / c

    L, D, MomA = mod.aeroForcesMulti(M, alfares, deltafres, cd, cl, cm, vres, obj.wingSurf, rho, obj.lRef, obj.M0, mres,
                             obj.m10, obj.xcg0, obj.xcgf, obj.pref, rep)
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = mod.thrustMulti(Press, mres, presv, spimpv, deltares, taures, rep, obj.psl, obj.M0, obj.m10,
                                 obj.lRef, obj.xcgf, obj.xcg0)
    T = np.asarray(T, dtype=np.float64)
    isp = np.asarray(isp, dtype=np.float64)
    Deps = np.asarray(Deps, dtype=np.float64)
    MomT = np.asarray(MomT, dtype=np.float64)

    MomTot = MomA + MomT

    r1 = hres + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = mres / np.exp((Dv1 + Dv2) / (obj.g0*isp))

    g0 = obj.g0
    eps = Deps + alfares
    g = []
    for alt in hres:
        if alt == 0:
            g.append(g0)
        else:
            g.append(obj.g0 * (obj.Re / (obj.Re + alt)) ** 2)
    #g = np.asarray(g, dtype=np.float64)
    # dynamic pressure

    q = 0.5 * rho * (vres ** 2)

    # accelerations

    ax = (T * np.cos(Deps) - D * np.cos(alfares) + L * np.sin(alfares)) / mres
    az = (T * np.sin(Deps) + D * np.sin(alfares) + L * np.cos(alfares)) / mres

    if flag_save:
        res = open(savedata_file + "res_{}_{}.txt".format(os.path.basename(__file__), timestr), "w")
        res.write("Number of NLP iterations: " + str(maxiter) + "\n" + "\n" + "v: " + str(
            vres) + "\n" + "Chi: " + str(np.rad2deg(chires))
            + "\n" + "Gamma: " + str(np.rad2deg(gammares)) + "\n" + "Teta: " + str(
            np.rad2deg(tetares)) + "\n" + "Lambda: "
            + str(np.rad2deg(lamres)) + "\n" + "Height: " + str(hres) + "\n" + "Mass: " + str(
            mres) + "\n" + "mf: " + str(mf) + "\n"
            + "Objective Function: " + str(-mf / obj.M0) + "\n" + "Alfa: "
            + str(np.rad2deg(alfares)) + "\n" + "Delta: " + str(deltares) + "\n" + "Delta f: " + str(
            np.rad2deg(deltafres)) + "\n"
            + "Tau: " + str(taures) + "\n" + "Eps: " + str(np.rad2deg(eps)) + "\n" + "Lift: "
            + str(L) + "\n" + "Drag: " + str(D) + "\n" + "Thrust: " + str(T) + "\n" + "Spimp: " + str(
            isp) + "\n" + "c: "
            + str(c) + "\n" + "Mach: " + str(M) + "\n" + "Time vector: " + str(timeTotal) + "\n" + "Press: " + str(
            Press) + "\n" + "Dens: " + str(rho) + "\n" + "Time elapsed during optimization: " + tformat)
        res.close()

        np.save(savedata_file + "v", vres)
        np.save(savedata_file + "chi", chires)
        np.save(savedata_file + "gamma", gammares)
        np.save(savedata_file + "teta", tetares)
        np.save(savedata_file + "lambda", lamres)
        np.save(savedata_file + "h", hres)
        np.save(savedata_file + "m", mres)
        np.save(savedata_file + "alfa", alfares)
        np.save(savedata_file + "delta", deltares)
        #np.save(savedata_file + "deltaf", deltaf)
        #np.save(savedata_file + "tau", tau)
        #np.save(savedata_file + "mu", mu)
        np.save(savedata_file + "timeTot", timeTotal)

    downrange = (vres ** 2) / g * np.sin(2 * gammares)

    plt.figure(0)
    plt.title("Velocity")
    plt.plot(timeTotal, vres)
    plt.grid()
    plt.ylabel("m/s")
    plt.xlabel("time [s]")
    if flag_save:
        plt.savefig(savefig_file + "velocity" + ".png")


    plt.figure(1)
    plt.title("Flight path angle \u03C7")
    plt.plot(timeTotal, np.rad2deg(chires))
    plt.grid()
    plt.ylabel("deg")
    plt.xlabel("time [s]")
    if flag_save:
        plt.savefig(savefig_file + "chi" + ".png")


    plt.figure(2)
    plt.title("Angle of climb \u03B3")
    plt.plot(timeTotal, np.rad2deg(gammares))
    plt.grid()
    plt.ylabel("deg")
    plt.xlabel("time [s]")
    if flag_save:
        plt.savefig(savefig_file + "gamma" + ".png")


    plt.figure(3)
    plt.title("Longitude \u03B8")
    plt.plot(timeTotal, np.rad2deg(tetares))
    plt.grid()
    plt.ylabel("deg")
    plt.xlabel("time [s]")
    if flag_save:
        plt.savefig(savefig_file + "theta" + ".png")


    plt.figure(4)
    plt.title("Latitude \u03BB")
    plt.plot(timeTotal, np.rad2deg(lamres))
    plt.grid()
    plt.ylabel("deg")
    plt.xlabel("time [s]")
    if flag_save:
        plt.savefig(savefig_file + "lambda" + ".png")


    plt.figure(5)
    plt.title("Flight angles")
    plt.plot(timeTotal, np.rad2deg(chires), color="g")
    plt.plot(timeTotal, np.rad2deg(gammares), color="b")
    plt.plot(timeTotal, np.rad2deg(tetares), color="r")
    plt.plot(timeTotal, np.rad2deg(lamres), color="k")
    plt.grid()
    plt.ylabel("deg")
    plt.xlabel("time [s]")
    plt.legend(["Chi", "Gamma", "Theta", "Lambda"], loc="best")
    if flag_save:
        plt.savefig(savefig_file + "angles" + ".png")


    plt.figure(6)
    plt.title("Altitude")
    plt.plot(timeTotal, hres / 1000)
    plt.grid()
    plt.ylabel("km")
    plt.xlabel("time [s]")
    if flag_save:
        plt.savefig(savefig_file + "altitude" + ".png")


    plt.figure(7)
    plt.title("Mass")
    plt.plot(timeTotal, mres)
    plt.grid()
    plt.ylabel("kg")
    plt.xlabel("time [s]")
    if flag_save:
        plt.savefig(savefig_file + "mass" + ".png")


    plt.figure(8)
    plt.title("Angle of attack \u03B1")
    plt.plot(tC, np.rad2deg(alfaCP), 'ro')
    plt.plot(timeTotal, np.rad2deg(alfares))
    plt.grid()
    plt.ylabel("deg")
    plt.xlabel("time [s]")
    plt.legend(["Control points"], loc="best")
    if flag_save:
        plt.savefig(savefig_file + "alpha" + ".png")


    plt.figure(9)
    plt.title("Throttles")
    plt.plot(timeTotal, deltares * 100, color='r')
    #plt.plot(timeTotal, taures * 100, color='k')
    plt.plot(tC, deltaCP * 100, 'ro')
    #plt.plot(tC, tauCP[i, :] * 100, 'ro')
    plt.grid()
    plt.ylabel("%")
    plt.xlabel("time [s]")
    plt.legend(["Delta", "Tau", "Control points"], loc="best")
    if flag_save:
        plt.savefig(savefig_file + "throttles" + ".png")


    '''plt.figure(10)
    plt.title("Body Flap deflection \u03B4")
    plt.plot(tC, np.rad2deg(deltafCP[i, :]), "ro")
    plt.plot(timeTotal, np.rad2deg(deltafres))
    plt.grid()
    plt.ylabel("deg")
    plt.xlabel("time [s]")
    plt.legend(["Control points"], loc="best")
    plt.axvline(time[i], color="k", alpha=0.5)
    if flag_save:
        plt.savefig(savefig_file + "deltaf" + ".png")

    plt.figure(11)
    plt.title("Bank angle profile \u03BC")
    plt.plot(tC, np.rad2deg(muCP[i, :]), "ro")
    plt.plot(timeTotal, np.rad2deg(mures))
    plt.grid()
    plt.ylabel("deg")
    plt.xlabel("time [s]")
    plt.legend(["Control points"], loc="best")
    plt.axvline(time[i], color="k", alpha=0.5)
    if flag_save:
        plt.savefig(savefig_file + "mu" + ".png")'''


    plt.figure(12)
    plt.title("Dynamic Pressure")
    plt.plot(timeTotal, q / 1000)
    plt.grid()
    plt.ylabel("kPa")
    plt.xlabel("time [s]")
    if flag_save:
        plt.savefig(savefig_file + "dynPress" + ".png")


    plt.figure(13)
    plt.title("Accelerations")
    plt.plot(timeTotal, ax, color='b')
    plt.plot(timeTotal, az, color='r')
    plt.grid()
    plt.ylabel("m/s^2")
    plt.xlabel("time [s]")
    plt.legend(["ax", "az"], loc="best")
    if flag_save:
        plt.savefig(savefig_file + "accelerations" + ".png")


    plt.figure(14)
    plt.title("Downrange")
    plt.plot(downrange / 1000, hres / 1000)
    plt.grid()
    plt.ylabel("km")
    plt.xlabel("km")
    if flag_save:
        plt.savefig(savefig_file + "downrange" + ".png")


    plt.figure(15)
    plt.title("Forces")
    plt.plot(timeTotal, T / 1000, color='r')
    plt.plot(timeTotal, L / 1000, color='b')
    plt.plot(timeTotal, D / 1000, color='k')
    plt.grid()
    plt.ylabel("kN")
    plt.xlabel("time [s]")
    plt.legend(["Thrust", "Lift", "Drag"], loc="best")
    if flag_save:
        plt.savefig(savefig_file + "forces" + ".png")


    plt.figure(16)
    plt.title("Mach")
    plt.plot(timeTotal, M)
    plt.grid()
    plt.xlabel("time [s]")
    if flag_save:
        plt.savefig(savefig_file + "mach" + ".png")


    plt.figure(17)
    plt.title("Total pitching Moment")
    plt.plot(timeTotal, MomTot / 1000, color='k')
    plt.grid()
    plt.axhline(5, 0, timeTotal[-1], color='r')
    plt.axhline(-5, 0, timeTotal[-1], color='r')
    plt.ylabel("kNm")
    plt.xlabel("time [s]")
    if flag_save:
        plt.savefig(savefig_file + "moment" + ".png")


    print("m before Ho : {0:.5f}".format(mres[-1]))
    print("mf          : {0:.5f}".format(mf[-1]))
    print("altitude Hohmann starts: {0:.5f}".format(hres[-1]))
    print("final time  : {}".format(tfin))


    plt.show(block=True)

